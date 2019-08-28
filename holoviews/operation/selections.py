import numpy as np
import param
from weakref import WeakValueDictionary

from holoviews import Overlay
from ..streams import SelectionExprStream, Params
from ..core.element import Element, HoloMap, Layout
from ..util import Dynamic
from ..core.options import Store


class link_selections(param.ParameterizedFunction):
    selection_expr = param.Parameter(default=None)
    color = param.Color(default='#DC143C')  # crimson
    unselected_color = param.Color(default="#99a6b2")  # LightSlateGray - 65%
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
        old_instance = link_selections._instances.get(self.selection_name, None)
        if old_instance is not None and old_instance is not self:
            old_instance.clear()

        link_selections._instances[self.selection_name] = self

        if isinstance(hvobj, Element):
            dmap = self._to_dmap_with_selection(hvobj)
        else:
            dmap = hvobj.map(
                lambda element: self._to_dmap_with_selection(element),
                specs=Element)

        # Collate overlays
        dmap = dmap.map(lambda overlay: overlay.collate(),
                        specs=Overlay)

        return dmap

    def _to_dmap_with_selection(self, hvobj):
        # Register element
        self._register_element(hvobj)

        # Dynamic operation function that returns element
        # with selected subset overlay

        # Convert to DynamicMap
        dmap = Dynamic(hvobj,
                       operation=_display_selection_fn,
                       streams=[self.param_stream])

        # Update dimension ranges
        if isinstance(hvobj, HoloMap):
            for d in hvobj.dimensions():
                dmap = dmap.redim.range(**{d.name: hvobj.range(d)})

        return dmap


def _display_selection_fn(element, **kwargs):
    if element._selection_display_mode == 'overlay':
        return _display_overlay_selection(element, **kwargs)
    elif element._selection_display_mode == 'color_list':
        return _display_color_list_selection(element, **kwargs)
    else:
        return element


def _get_color_property(element, color):
    element_name = type(element).name
    if Store.current_backend == 'bokeh':
        if element_name == 'Violin':
            return {"violin_fill_color": color}
        elif element_name == 'Bivariate':
            return {"cmap": [color]}

    return {"color": color}


def _display_overlay_selection(element, selection_expr, color, unselected_color, **kwargs):
    """
    Display a selection on an element by overlaying a subset of the element
    on top of itself
    """
    if Store.current_backend == 'bokeh':
        backend_options = Store.options(backend='bokeh')
        style_options = backend_options[(type(element).name,)]['style']

        def alpha_opts(alpha):
            options = dict()

            for opt_name in style_options.allowed_keywords:
                if 'alpha' in opt_name:
                    options[opt_name] = alpha

            return options

        overlay_alpha = 1.0 if selection_expr else 0.0
        return (element.options(
            **_get_color_property(element, unselected_color),
            **alpha_opts(0.9)) *
                element.select(selection_expr).options(
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

        if selection_expr:
            overlay_opts = {}
        else:
            overlay_opts = {'visible': False}

        return (element.options(color=unselected_color, **shared_opts) *
                element.select(selection_expr).options(
                    color=color,
                    **overlay_opts,
                    **shared_opts
                ))


def _display_color_list_selection(element, selection_expr, color, unselected_color, **kwargs):

    if Store.current_backend == 'plotly':
        if not selection_expr:
            colors = [unselected_color] * len(element.dimension_values(0))
            return element.options(color=colors)
        else:
            selection_mask = np.array(selection_expr.apply(element), dtype='int8')
            clrs = np.array([unselected_color, color])
            colors = clrs[selection_mask]
            return element.options(color=colors)
    else:
        return element
