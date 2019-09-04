import numpy as np
import param
from weakref import WeakValueDictionary

from param.parameterized import bothmethod

from holoviews import Overlay
from holoviews.core import OperationCallable
from ..streams import SelectionExpr, Params, Stream
from ..core.element import Element, Layout
from ..util import Dynamic, DynamicMap
from ..core.options import Store
from ..plotting.util import initialize_dynamic, linear_gradient

_Cmap = Stream.define('Cmap', cmap=[])
_Alpha = Stream.define('Alpha', alpha=1.0)
_Exprs = Stream.define('Exprs', exprs=[])
_Colors = Stream.define('Colors', colors=[])


class link_selections(param.ParameterizedFunction):
    selection_expr = param.Parameter(default=None)
    unselected_color = param.Color(default="#99a6b2")  # LightSlateGray - 65%
    selected_color = param.Color(default="#DC143C")  # Crimson

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super(link_selections, self_or_cls).instance(**params)

        # Init private properties
        inst._selection_expr_streams = []

        # Colors stream
        inst._colors_stream = _Colors(
            colors=[inst.unselected_color, inst.selected_color]
        )

        # Cmap streams
        inst._cmap_streams = [
            _Cmap(cmap=inst.unselected_cmap),
            _Cmap(cmap=inst.selected_cmap),
        ]

        def update_colors(*_):
            inst._colors_stream.event(
                colors=[inst.unselected_color, inst.selected_color]
            )
            inst._cmap_streams[0].event(cmap=inst.unselected_cmap)
            inst._cmap_streams[1].event(cmap=inst.selected_cmap)

        inst.param.watch(
            update_colors,
            parameter_names=['unselected_color', 'selected_color']
        )

        # Exprs stream
        inst._exprs_stream = _Exprs(exprs=[True, None])

        def update_exprs(*_):
            inst._exprs_stream.event(exprs=[True, inst.selection_expr])

        inst.param.watch(
            update_exprs,
            parameter_names=['selection_expr']
        )

        # Alpha streams
        inst._alpha_streams = [
            _Alpha(alpha=255),
            _Alpha(alpha=inst._selected_alpha),
        ]

        def update_alphas(*_):
            inst._alpha_streams[1].event(alpha=inst._selected_alpha)

        inst.param.watch(update_alphas, parameter_names=['selection_expr'])
        return inst

    @property
    def unselected_cmap(self):
        return _color_to_cmap(self.unselected_color)

    @property
    def selected_cmap(self):
        return _color_to_cmap(self.selected_color)

    @property
    def _selected_alpha(self):
        if self.selection_expr:
            return 255
        else:
            return 0

    def _register_element(self, element):
        expr_stream = SelectionExpr(source=element)

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
        elif isinstance(hvobj, (Layout, Overlay)):
            new_hvobj = hvobj.clone(shared_data=False)
            for k, v in hvobj.items():
                new_hvobj[k] = self._selection_transform(
                    v, operations
                )

            # collate if available. Needed for Overlay
            try:
                new_hvobj = new_hvobj.collate()
            except AttributeError:
                pass

            return new_hvobj
        else:
            # Unsupported object
            return hvobj

    def _build_overlay_selection(self, element, operations=()):
        base_layer = Dynamic(
            element,
            operation=_build_layer_callback(0),
            streams=[self._colors_stream, self._exprs_stream]
        )
        selection_layer = Dynamic(
            element,
            operation=_build_layer_callback(1),
            streams=[self._colors_stream, self._exprs_stream]
        )

        # Wrap in operations
        for op in operations:
            if 'cmap' in op.param:
                # Add in the selection color as cmap for operation
                base_op = op.instance(
                    streams=op.streams + [self._cmap_streams[0]]
                )

                select_streams = op.streams + [self._cmap_streams[1]]

                if 'alpha' in op.param:
                    select_streams += [self._alpha_streams[1]]

                select_op = op.instance(streams=select_streams)

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

        def _build_selection(el, colors, exprs, **_):

            selection_exprs = exprs[1:]
            unselected_color = colors[0]
            selected_colors = colors[1:]
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
            streams=[self._colors_stream, self._exprs_stream]
        )

        # Wrap in operations
        for op in operations:
            dmap = op(dmap)

        return dmap

    def _apply_op_with_color(self, hvobj, op, layer_number):
        def _color_op_fn(element, colors, **_):
            op_kwargs = _build_op_color_kwargs(op, colors[layer_number])
            return op(element, **op_kwargs)

        if _build_op_color_kwargs(op, 'dummy'):
            return Dynamic(hvobj,
                           operation=_color_op_fn,
                           streams=[self._colors_stream])
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
    def _build_layer(element, colors, exprs, **_):
        layer_element = _build_element_layer(
            element, colors[layer_number], exprs[layer_number]
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
