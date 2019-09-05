from __future__ import absolute_import
from holoviews.operation.selections import OverlaySelectionDisplay
from holoviews.core.options import Store


class PlotlyOverlaySelectionDisplay(OverlaySelectionDisplay):

    def _build_element_layer(
            self, element, layer_color, selection_expr=True
    ):
        element, visible = self._select(element, selection_expr)

        backend_options = Store.options(backend='plotly')
        style_options = backend_options[(type(element).name,)]['style']

        if 'selectedpoints' in style_options.allowed_keywords:
            shared_opts = dict(selectedpoints=False)
        else:
            shared_opts = dict()

        layer_element = element.options(
            visible=visible,
            **self._get_color_kwarg(layer_color),
            **shared_opts
        )

        return layer_element
