import numpy as np
import param
from weakref import WeakValueDictionary

from holoviews import Overlay
from holoviews.core import OperationCallable
from ..streams import SelectionExprStream, Params, Stream
from ..core.element import Element, Layout
from ..util import Dynamic, DynamicMap
from ..core.options import Store
from ..plotting.util import initialize_dynamic, linear_gradient

_UnselectedCmap = Stream.define('UnselectedCmap', cmap=[])
_SelectedCmap = Stream.define('SelectedCmap', cmap=[])
_Alpha = Stream.define('Alpha', alpha=1.0)


class link_selections(param.ParameterizedFunction):
    selection_expr = param.Parameter(default=None)
    unselected_color = param.Color(default="#99a6b2")  # LightSlateGray - 65%
    selected_color = param.Color(default="#DC143C")  # Crimson

    @property
    def _selection_expr_streams(self):
        try:
            return self.__selection_expr_streams
        except AttributeError:
            self.__selection_expr_streams = []
            return self.__selection_expr_streams

    @property
    def param_stream(self):
        try:
            return self._param_stream
        except AttributeError:
            self._param_stream = Params(
                self, parameters=[
                    'selection_expr',
                    'unselected_color',
                    'selected_color'
                ]
            )
            return self._param_stream

    @property
    @param.depends('unselected_color')
    def unselected_cmap(self):
        return _color_to_cmap(self.unselected_color)

    @property
    @param.depends('selected_color')
    def selected_cmap(self):
        return _color_to_cmap(self.selected_color)

    @property
    def _unselected_cmap_stream(self):
        try:
            return self.__unselected_cmap_stream
        except AttributeError:
            self.__unselected_cmap_stream = _UnselectedCmap(
                cmap=self.unselected_cmap
            )

            def _update_cmap(*_):
                self.__unselected_cmap_stream.event(cmap=self.unselected_cmap)

            self.param.watch(_update_cmap, parameter_names=['unselected_color'])

            return self.__unselected_cmap_stream

    @property
    def _selected_cmap_stream(self):
        try:
            return self.__selected_cmap_stream
        except AttributeError:
            self.__selected_cmap_stream = _SelectedCmap(
                cmap=self.selected_cmap
            )

            def _update_cmap(*_):
                self.__selected_cmap_stream.event(cmap=self.selected_cmap)

            self.param.watch(_update_cmap, parameter_names=['selected_color'])

            return self.__selected_cmap_stream

    @property
    def _selected_alpha(self):
        if self.selection_expr:
            return 255
        else:
            return 0

    @property
    def _selected_alpha_stream(self):
        try:
            return self.__selected_alpha_stream
        except AttributeError:
            self.__selected_alpha_stream = _Alpha(
                alpha=self._selected_alpha
            )

            def _update_alpha(*_):
                self.__selected_alpha_stream.event(alpha=self._selected_alpha)

            self.param.watch(_update_alpha, parameter_names=['selection_expr'])

            return self.__selected_alpha_stream

    def _register_element(self, element):
        expr_stream = SelectionExprStream(source=element)

        def _update_expr(selection_expr, bbox):
            if selection_expr:
                self.selection_expr = selection_expr

        expr_stream.add_subscriber(_update_expr)

        self._selection_expr_streams.append(expr_stream)

    def __del__(self):
        self.clear()

    def clear(self):
        for stream in self._selection_expr_streams:
            stream.source = None
            stream.clear()
        self._selection_expr_streams.clear()
        self.param_stream.clear()
        self.selection_expr = None

    def __call__(self, hvobj, **kwargs):
        ## Apply params
        self.param.set_param(**kwargs)

        # Perform transform
        hvobj_selection = self._selection_transform(hvobj.clone(link=False))

        return hvobj_selection

    def _selection_transform(
            self,
            hvobj,
            operations=(),
    ):
        # hvobj: DynamicMap or Element (not Layout)
        if isinstance(hvobj, DynamicMap):
            initialize_dynamic(hvobj)

            if (isinstance(hvobj.callback, OperationCallable) and
                    len(hvobj.callback.inputs) == 1):

                child_hvobj = hvobj.callback.inputs[0]
                next_op = hvobj.callback.operation
                new_operations = (next_op,) + operations

                # Recurse on child with added operation
                return self._selection_transform(
                    hvobj=child_hvobj,
                    operations=new_operations,
                )
            elif hvobj.type == Overlay and not hvobj.streams:
                # Process overlay inputs individually and then overlay again
                overlay_elements = hvobj.callback.inputs
                new_hvobj = self._selection_transform(overlay_elements[0])
                for overlay_element in overlay_elements[1:]:
                    new_hvobj = new_hvobj * self._selection_transform(overlay_element)

                return new_hvobj
            else:
                # This is a DynamicMap that we don't know how to recurse into.
                # TODO: see if we can transform the output
                return hvobj

        elif isinstance(hvobj, Element):
            element = hvobj.clone(link=False)

            # Register element to receive selection expression callbacks
            self._register_element(element)

            # Build appropriate selection callback for element type
            if element._selection_display_mode == 'overlay':
                # return element/dynamic map?
                dmap = self._build_overlay_selection(element, operations)
            elif element._selection_display_mode == 'color_list':
                dmap = self._build_colorlist_selection(
                    element, operations
                )
            else:
                # Unsupported element
                return hvobj

            return dmap
        elif isinstance(hvobj, Layout):
            new_hvobj = hvobj.clone(shared_data=False)
            for k, v in hvobj.items():
                new_hvobj[k] = self._selection_transform(
                    v, operations
                )
            return new_hvobj
        else:
            # Unsupported object
            return hvobj

    def _build_overlay_selection(self, element, operations=()):
        base_layer = Dynamic(
            element,
            operation=_build_layer_callback(0),
            streams=[self.param_stream]
        )
        selection_layer = Dynamic(
            element,
            operation=_build_layer_callback(1),
            streams=[self.param_stream]
        )

        # Wrap in operations
        for op in operations:
            if 'cmap' in op.param:
                # Add in the selection color as cmap for operation
                base_op = op.instance(
                    streams=op.streams + [self._unselected_cmap_stream],
                )

                select_streams = op.streams + [self._selected_cmap_stream]
                if 'alpha' in op.param:
                    select_streams += [self._selected_alpha_stream]

                select_op = op.instance(
                    streams=select_streams,
                )

                base_layer = base_op(base_layer)
                selection_layer = select_op(selection_layer)
            else:
                base_layer = op(base_layer)
                selection_layer = op(selection_layer)

        # Overlay
        return base_layer * selection_layer

    def _build_colorlist_selection(self, element, operations=()):
        """
        Build selections on an element by overlaying subsets of the element
        on top of itself
        """

        def _build_selection(
                el,
                unselected_color,  # from param stream
                selected_color,  # from param stream
                selection_expr,  # from param stream
                **_,
        ):

            selection_exprs = [selection_expr]
            selected_colors = [selected_color]
            if Store.current_backend == 'plotly':
                n = len(el.dimension_values(0))

                if not any(selection_exprs):
                    colors = [unselected_color] * n
                    return el.options(color=colors)
                else:
                    clrs = np.array(
                        [unselected_color] + list(selected_colors))

                    color_inds = np.zeros(n, dtype='int8')

                    for i, expr, color in zip(
                            range(1, len(clrs)),
                            selection_exprs,
                            selected_colors
                    ):
                        color_inds[expr.apply(el)] = i

                    colors = clrs[color_inds]

                    return el.options(color=colors)
            else:
                return el

        dmap = Dynamic(
            element,
            operation=_build_selection,
            streams=[self.param_stream]
        )

        # Wrap in operations
        for op in operations:
            dmap = op(dmap)

        return dmap

    def _apply_op_with_color(self, hvobj, op, layer_number):
        def _color_op_fn(
                element,
                unselected_color,  # from param stream
                selected_color,  # from param stream
                **_,
        ):
            if layer_number == 0:
                color = unselected_color
            else:
                color = selected_color

            op_kwargs = _build_op_color_kwargs(op, color)
            return op(element, **op_kwargs)

        if _build_op_color_kwargs(op, 'dummy'):
            return Dynamic(hvobj,
                           operation=_color_op_fn,
                           streams=[self.param_stream])
        else:
            return op(hvobj)


def _color_to_cmap(color):
    """
    Create a light to dark cmap list from a base color
    """
    # Lighten start color by interpolating toward white
    start_color = linear_gradient("#ffffff", color, 7)[2]

    # Darken end color by interpolating toward black
    end_color = linear_gradient(color, "#000000", 7)[2]
    return [start_color, end_color]


def _build_op_color_kwargs(op, color):
    kwargs = {}

    if color:
        if 'cmap' in op.param:
            kwargs['cmap'] = [color]

        if 'color' in op.param:
            kwargs['color'] = color

    return kwargs


def _get_color_property(element, color):
    element_name = type(element).name
    if Store.current_backend == 'bokeh':
        if element_name == 'Violin':
            return {"violin_fill_color": color}
        elif element_name == 'Bivariate':
            return {"cmap": [color]}

    return {"color": color}


def _build_layer_callback(layer_number):
    def _build_layer(
            element,
            unselected_color,  # from param stream
            selected_color,  # from param stream
            selection_expr,  # from param stream
            **_,
    ):
        if layer_number == 0:
            layer_color = unselected_color
            selection_expr = True
        else:
            layer_color = selected_color

        layer_element = _build_element_layer(
            element, layer_color, selection_expr
        )

        return layer_element

    return _build_layer


def _build_element_layer(
        element, layer_color, selection_expr=True
):
    from ..util.transform import dim
    if isinstance(selection_expr, dim):
        element = element.select(selection_expr=selection_expr)
        visible = True
    else:
        visible = bool(selection_expr)

    if Store.current_backend == 'bokeh':
        backend_options = Store.options(backend='bokeh')
        style_options = backend_options[(type(element).name,)]['style']

        def alpha_opts(alpha):
            options = dict()

            for opt_name in style_options.allowed_keywords:
                if 'alpha' in opt_name:
                    options[opt_name] = alpha

            return options

        layer_alpha = 1.0 if visible else 0.0
        layer_element = element.options(
            **_get_color_property(element, layer_color),
            **alpha_opts(layer_alpha)
        )

        return layer_element

    elif Store.current_backend == 'plotly':
        backend_options = Store.options(backend='plotly')
        style_options = backend_options[(type(element).name,)]['style']

        if 'selectedpoints' in style_options.allowed_keywords:
            shared_opts = dict(selectedpoints=False)
        else:
            shared_opts = dict()

        layer_element = element.options(
            visible=visible,
            **_get_color_property(element, layer_color),
            **shared_opts
        )

        return layer_element
    else:
        raise ValueError("Unsupported backend: %s" % Store.current_backend)
