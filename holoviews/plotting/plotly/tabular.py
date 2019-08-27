from __future__ import absolute_import, division, unicode_literals

import param

from .element import ElementPlot


class TablePlot(ElementPlot):

    height = param.Number(default=400)

    width = param.Number(default=400)

    trace_kwargs = {'type': 'table'}

    style_opts = ['line_color', 'color', 'align', 'font', 'cell_height']

    _style_key = 'cells'

    def get_data(self, element, ranges, style):
        header = dict(values=[d.pprint_label for d in element.dimensions()])
        cells = dict(values=[[d.pprint_value(v) for v in element.dimension_values(d)]
                              for d in element.dimensions()])

        return [{'header': header, 'cells': cells}]

    def graph_options(self, element, ranges, style):
        opts = super(TablePlot, self).graph_options(element, ranges, style)

        # Transpose fill_color array so values apply by rows not column
        if 'color' in opts.get('cells', {}):
            opts['cells']['fill_color'] = [opts['cells'].pop('color')]

        if 'line_color' in opts.get('cells', {}):
            opts['cells']['line_color'] = [opts['cells']['line_color']]

        return opts

    def init_layout(self, key, element, ranges):
        return dict(width=self.width, height=self.height,
                    title=self._format_title(key, separator=' '),
                    plot_bgcolor=self.bgcolor)
