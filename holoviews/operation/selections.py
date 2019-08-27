import param
from weakref import WeakValueDictionary
from ..streams import SelectionExprStream, Params
from ..core.element import Element, HoloMap, Layout
from ..util import Dynamic
from ..core.options import Store


class link_selections(param.ParameterizedFunction):
    selection_expr = param.Parameter(default=None)
    color = param.Color(default='#DC143C')  # crimson
    unselected_color = param.Color(default='#B0C4DE')  # lightsteelblue
    selection_name = param.String(default=None, constant=True)
    overwrite = param.Boolean(default=None)

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

        # if self.overwrite is not None:
        #     overwrite = self.overwrite
        # else:
        #     overwrite = self.name is None

        old_instance = link_selections._instances.get(self.selection_name, None)
        if old_instance is not None and old_instance is not self:
            old_instance.clear()

        link_selections._instances[self.selection_name] = self

        if isinstance(hvobj, Layout):
            dmap = hvobj.map(
                lambda element: self._to_dmap_with_selection(element),
                specs=Element)
        else:
            dmap = self._to_dmap_with_selection(hvobj)

        return dmap

    def _to_dmap_with_selection(self, hvobj):
        # Register element
        self._register_element(hvobj)

        # Dynamic operation function that returns element
        # with selected subset overlay

        # Convert to DynamicMap
        dmap = Dynamic(hvobj,
                       operation=_overlay_selection_fn,
                       streams=[self.param_stream])

        # Update dimension ranges
        if isinstance(hvobj, HoloMap):
            for d in hvobj.dimensions():
                dmap = dmap.redim.range(**{d.name: hvobj.range(d)})

        return dmap


def _overlay_selection_fn(element, selection_expr, color, unselected_color, **_):
    if Store.current_backend == 'bokeh':
        def alpha_opts(alpha):
            return dict(selection_alpha=alpha,
                        nonselection_alpha=alpha,
                        alpha=alpha)

        overlay_alpha = 1.0 if selection_expr else 0.0
        return (element.options(line_alpha=1.0,
                                color=unselected_color,
                                **alpha_opts(0.9)) *
                element.select(selection_expr).options(
                    color=color, **alpha_opts(overlay_alpha)
                ))

    elif Store.current_backend == 'plotly':
        shared_opts = dict(selectedpoints=False)
        overlay_alpha = 1.0 if selection_expr else 0.0
        return (element.options(color=unselected_color, **shared_opts) *
                element.select(selection_expr).options(
                    color=color,
                    opacity=overlay_alpha,
                    **shared_opts
                ))
