from holoviews.element.comparison import ComparisonTestCase
import pandas as pd
from holoviews import Dataset, Curve, Dimension, Scatter, Histogram
import dask.dataframe as dd

class DatasetPropertyTestCase(ComparisonTestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1, 1, 3, 3, 2, 2, 0, 0],
            'b': [10, 20, 30, 40, 10, 20, 30, 40],
            'c': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'd': [-1, -2, -3, -4, -5, -6, -7, -8]
        })

        self.ds = Dataset(
            self.df,
            kdims=[
                Dimension('a', label="The a Column"),
                Dimension('b', label="The b Column"),
                Dimension('c', label="The c Column"),
                Dimension('d', label="The d Column"),
            ]
        )


class ConstructorTestCase(DatasetPropertyTestCase):
    def test_constructors(self):
        expected = Dataset(self.df)
        self.assertIs(expected, expected.dataset)

        element = Curve(self.df)
        self.assertEqual(element.dataset, expected)


class ToTestCase(DatasetPropertyTestCase):
    def test_to_element(self):
        curve = self.ds.to(Curve, 'a', 'b', groupby=[])
        self.assertEqual(curve.dataset, self.ds)

        scatter = curve.to(Scatter)
        self.assertEqual(scatter.dataset, self.ds)

    def test_to_holomap(self):
        curve_hmap = self.ds.to(Curve, 'a', 'b', groupby=['c'])

        # Check HoloMap element datasets
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[(v,)]
            self.assertEqual(
                curve.dataset, self.ds.select(c=v)
            )

    def test_to_holomap_dask(self):
        ddf = dd.from_pandas(self.df, npartitions=2)
        dds = Dataset(
            ddf,
            kdims=[
                Dimension('a', label="The a Column"),
                Dimension('b', label="The b Column"),
                Dimension('c', label="The c Column"),
                Dimension('d', label="The d Column"),
            ]
        )

        curve_hmap = dds.to(Curve, 'a', 'b', groupby=['c'])

        # Check HoloMap element datasets
        for v in self.df.c.drop_duplicates():
            curve = curve_hmap.data[(v,)]
            self.assertEqual(
                curve.dataset, self.ds.select(c=v)
            )


class CloneTestCase(DatasetPropertyTestCase):
    def test_clone(self):
        # Dataset
        self.assertEqual(self.ds.clone().dataset, self.ds)

        # Curve
        self.assertEqual(
            self.ds.to.curve('a', 'b', groupby=[]).clone().dataset,
            self.ds
        )


class ReindexTestCase(DatasetPropertyTestCase):
    def test_reindex_dataset(self):
        ds_ab = self.ds.reindex(kdims=['a'], vdims=['b'])
        self.assertEqual(ds_ab.dataset, self.ds)

    def test_double_reindex_dataset(self):
        ds_abc = self.ds.reindex(kdims=['a'], vdims=['b', 'c'])
        ds_ab = ds_abc.reindex(kdims=['a'], vdims=['b'])
        self.assertEqual(ds_ab.dataset, self.ds)

    def test_reindex_curve(self):
        curve_ab = self.ds.to(Curve, 'a', 'b', groupby=[])
        curve_ba = curve_ab.reindex(kdims='b', vdims='a')
        self.assertEqual(curve_ab.dataset, self.ds)
        self.assertEqual(curve_ba.dataset, self.ds)

    def test_double_reindex_curve(self):
        curve_abc = self.ds.to(Curve, 'a', ['b', 'c'], groupby=[])
        curve_ab = curve_abc.reindex(kdims='a', vdims='b')
        curve_ba = curve_ab.reindex(kdims='b', vdims='a')
        self.assertEqual(curve_ab.dataset, self.ds)
        self.assertEqual(curve_ba.dataset, self.ds)