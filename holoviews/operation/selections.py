import numpy as np
import param
from weakref import WeakValueDictionary

from holoviews import Overlay
from holoviews.core import OperationCallable
from ..streams import SelectionExprStream, Params
from ..core.element import Element, HoloMap, Layout
from ..util import Dynamic, DynamicMap
from ..core.options import Store
from ..plotting.util import initialize_dynamic


class link_selections(param.ParameterizedFunction):
    selection_expr = param.Parameter(default=None)
    unselected_color = param.Color(default="#99a6b2")  # LightSlateGray - 65%
    selection_color = param.Color(default="#DC143C")  # Crimson
    selection_name = param.String(default=None, constant=True)
    _instances = WeakValueDictionary()

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
            self._param_stream = Params(self)
            return self._param_stream

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

    def __call__(self, hvobj):
        # Handle clearing existing selection with same name
        old_instance = link_selections._instances.get(
            self.selection_name, None
        )
        if old_instance is not None and old_instance is not self:
            old_instance.clear()

        link_selections._instances[self.selection_name] = self

        # Perform transform
        hvobj_selection = self._selection_transform(hvobj)

        return hvobj_selection

    # def _to_dmap_with_selection(self, hvobj):
    #
    #     # hvobj: DynamicMap or Element
    #     dmap = self._build_selection_dynamic_map(hvobj)
    #
    #     if dmap is not None:
    #         # Update dimension ranges
    #         if isinstance(hvobj, HoloMap):
    #             for d in hvobj.dimensions():
    #                 dmap = dmap.redim.range(**{d.name: hvobj.range(d)})
    #
    #         return dmap
    #     else:
    #         # Unsupported object for selection, return as-is
    #         return hvobj

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
                return self._build_selection_dynamic_map(
                    hvobj=child_hvobj,
                    operations=new_operations,
                )
            else:
                # This is a DynamicMap that we don't know how to recurse into.
                # TODO: see if we can transform the output
                return hvobj

        elif isinstance(hvobj, Element):
            element = hvobj.clone()

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
        # elif hvobj._deep_indexable:
        #     new_hvobj = hvobj.clone(shared_data=False)
        #     for k, v in new_hvobj.items():
        #         new_hvobj[k] = self._selection_transform(
        #             v, operations
        #         )
        #
        #     try:
        #         # Collate if container type supports it
        #         new_hvobj = new_hvobj.collate()
        #     except AttributeError:
        #         pass
        #
        #     return new_hvobj
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
                selection_color,  # from param stream
                selection_expr,  # from param stream
                **_,
        ):

            selection_exprs = [selection_expr]
            selection_colors = [selection_color]
            if Store.current_backend == 'plotly':
                n = len(el.dimension_values(0))

                if not any(selection_exprs):
                    colors = [unselected_color] * n
                    return el.options(color=colors)
                else:
                    clrs = np.array(
                        [unselected_color] + list(selection_colors))

                    color_inds = np.zeros(n, dtype='int8')

                    for i, expr, color in zip(
                            range(1, len(clrs)),
                            selection_exprs,
                            selection_colors
                    ):
                        color_inds[expr.apply(el)] = i

                    colors = clrs[color_inds]

                    return el.options(color=colors)
            else:
                return el

        selection = Dynamic(
            element,
            operation=_build_selection,
            streams=[self.param_stream]
        )

        # Wrap in operations
        for op in operations:
            selection = op(selection)

        return selection



# def _build_overlay_callback():
#     """
#     Build selections on an element by overlaying subsets of the element
#     on top of itself
#     """
#     def _build_selection(
#             element,
#             unselected_color,  # from param stream
#             selection_color,  # from param stream
#             selection_expr,  # from param stream
#             **_,
#     ):
#         selection_colors = [selection_color]
#         base_element, overlay_elements = _build_element_layers(
#             element,
#             unselected_color=unselected_color,
#             selection_colors=selection_colors,
#             selection_exprs=[selection_expr]
#         )
#
#         result = base_element
#         for overlay_element, color in zip(overlay_elements, selection_colors):
#             result = result * overlay_element
#
#         return result
#
#
#     return _build_selection
#
#
# def _build_colorlist_callback():
#     """
#     Build selections on an element by overlaying subsets of the element
#     on top of itself
#     """
#     def _build_selection(
#             element,
#             unselected_color,  # from param stream
#             selection_color,  # from param stream
#             selection_expr,  # from param stream
#             **_,
#     ):
#
#         selection_exprs = [selection_expr]
#         selection_colors = [selection_color]
#         if Store.current_backend == 'plotly':
#             n = len(element.dimension_values(0))
#
#             if not any(selection_exprs):
#                 colors = [unselected_color] * n
#                 return element.options(color=colors)
#             else:
#                 clrs = np.array([unselected_color] + list(selection_colors))
#
#                 color_inds = np.zeros(n, dtype='int8')
#
#                 for i, expr, color in zip(
#                         range(1, len(clrs)),
#                         selection_exprs,
#                         selection_colors
#                 ):
#                     color_inds[expr.apply(element)] = i
#
#                 colors = clrs[color_inds]
#
#                 return element.options(color=colors)
#         else:
#             return element
#
#     return _build_selection


def _build_apply_ops(operations, color=None):
    # Build function that applies a sequence of operations to an input element
    # Also, apply any of the args in kwargs that are compatible with each
    # operation
    def fn(v):
        for op in operations:
            op_kwargs = _build_op_color_kwargs(op, color)

            v = op(v, **op_kwargs)
        return v
    return fn


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
            selection_color,  # from param stream
            selection_expr,  # from param stream
            **_,
    ):
        if layer_number == 0:
            layer_color = unselected_color
            selection_expr = True
        else:
            layer_color = selection_color

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
        element = element.select(selection_expr)
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


def _build_element_layers(
        element, unselected_color, selection_colors, selection_exprs
):
    if Store.current_backend == 'bokeh':
        backend_options = Store.options(backend='bokeh')
        style_options = backend_options[(type(element).name,)]['style']

        def alpha_opts(alpha):
            options = dict()

            for opt_name in style_options.allowed_keywords:
                if 'alpha' in opt_name:
                    options[opt_name] = alpha

            return options

        base_element = element.options(
            **_get_color_property(element, unselected_color),
            **alpha_opts(0.9)
        )

        overlay_elements = []
        for color, expr in zip(selection_colors, selection_exprs):
            overlay_alpha = 1.0 if expr else 0.0
            overlay_elements.append(element.select(expr).options(
                **_get_color_property(element, color),
                **alpha_opts(overlay_alpha)
            ))

    elif Store.current_backend == 'plotly':
        backend_options = Store.options(backend='plotly')
        style_options = backend_options[(type(element).name,)]['style']

        if 'selectedpoints' in style_options.allowed_keywords:
            shared_opts = dict(selectedpoints=False)
        else:
            shared_opts = dict()


        base_element = element.options(
            **_get_color_property(element, unselected_color),
            **shared_opts
        )

        overlay_elements = []
        for color, expr in zip(selection_colors, selection_exprs):
            if expr:
                overlay_opts = {}
            else:
                overlay_opts = {'visible': False}

            overlay_elements.append(element.select(expr).options(
                **_get_color_property(element, color),
                **shared_opts,
                **overlay_opts
            ))
    else:
        raise ValueError("Unsupported backend: %s" % Store.current_backend)

    return base_element, overlay_elements


def _maybe_map_ops(element, color=None, operations=None):
    if operations:
        return element.map(_build_apply_ops(operations, color))
    else:
        return element
