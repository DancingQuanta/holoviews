from __future__ import absolute_import
from holoviews.operation.selections import OverlaySelectionDisplay
from holoviews.core.options import Store


class BokehOverlaySelectionDisplay(OverlaySelectionDisplay):

    def _build_element_layer(
            self, element, layer_color, selection_expr=True
    ):
        element, visible = self._select(element, selection_expr)

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
            **self._get_color_kwarg(layer_color),
            **alpha_opts(layer_alpha)
        )

        return layer_element
