"""Configure superframe. Inherit DataArray for fast access else DataFrame."""
from nova.electromagnetic.metamethod import MetaMethod
from nova.electromagnetic.indexer import Indexer
from nova.electromagnetic.energize import Energize
from nova.electromagnetic.multipoint import MultiPoint
from nova.electromagnetic.polygon import Polygon


class SubSpaceError(IndexError):
    """Prevent direct access to frame's subspace variables."""

    def __init__(self, name, col):
        super().__init__(
            f'{name} access is restricted for subspace attributes. '
            f'Use frame.subspace.{name}[:, {col}] = *.\n\n'
            'Lock may be overridden via the following context manager '
            'but subspace will still overwrite (Cavieat Usor):\n'
            'with frame.metaframe.setlock(None):\n'
            f'    frame.{name}[:, {col}] = *')


class FrameMixin(ArrayMixin):
    """Extend set/getitem methods for loc, iloc, at, and iat accessors."""

    def __setitem__(self, key, value):
        """Raise error when subspace variable is set directly from frame."""
        col = self.obj._get_col(key)
        value = self.obj._format_value(col, value)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                raise SubSpaceError(self.name, col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        """Refresh subspace items prior to return."""
        col = self.obj._get_col(key)
        if self.obj.in_field(col, 'subspace'):
            if self.obj.metaframe.lock('subspace') is True:
                self.obj.set_frame(col)
        if self.obj.in_field(col, 'energize'):
            if self.obj.metaframe.lock('energize') is False:
                return self.obj.energize._get_item(super(), key)
        return super().__getitem__(key)


class FrameIndexer(ArrayIndexer):

    @property
    def loc_mixin(self):
        """Return LocIndexer mixins."""
        return FrameMixin


class SuperFrame(FrameIndexer, DataArray):
    """
    Extend pandas.DataFrame.

    - Manage Frame metadata (metaarray, metaframe).
    - Add boolean methods (add_frame, drop_frame...).

    """

    def __init__(self,
                 data=None,
                 index: Collection[Any] = None,
                 columns: Collection[Any] = None,
                 attrs: dict[str, Collection[Any]] = None,
                 **metadata: dict[str, Collection[Any]]):
        super().__init__(data, index, columns, attrs, **metadata)
        super().__init__(data, index, columns)
        self.update_metadata(data, columns, attrs, metadata)
        self.update_attrs()
        self.update_index()
        self.update_columns()
        self.init_attrs()

        # _extract_attrs
        #   self.attrs['indexer'] = Indexer(FrameMixin)  # init loc indexer


    def __repr__(self):
        """Propagate frame subspace variables prior to display."""
        self.update_frame()
        return super().__repr__()

    def update_attrs(self):
        """Extract frame attrs from data and update."""
        self.attrs['energize'] = Energize(self)
        self.attrs['multipoint'] = MultiPoint(self)
        self.attrs['polygon'] = Polygon(self)

    def init_attrs(self):
        """Initialize attributes (metamethods)."""
        for attr in self.attrs:
            attribute = self.attrs[attr]
            if isinstance(attribute, MetaMethod):
                if attribute.generate:
                    attribute.initialize()

    def __getitem__(self, key):
        """Extend DataFrame.__getitem__. (frame['*'])."""
        col = self._get_col(key)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__getitem__(col)
            if self.metaframe.lock('subspace') is False:
                self.set_frame(col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                return self.energize._get_item(super(), key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Check lock. Extend DataFrame.__setitem__. (frame['*'] = *)."""
        col = self._get_col(key)
        value = self._format_value(col, value)
        if self.in_field(col, 'subspace'):
            if self.metaframe.lock('subspace') is True:
                return self.subspace.__setitem__(key, value)
            if self.metaframe.lock('subspace') is False:
                raise SubSpaceError('setitem', col)
        if self.in_field(col, 'energize'):
            if self.metaframe.lock('energize') is False:
                return self.energize._set_item(super(), key, value)
        return super().__setitem__(key, value)

    def _build_data(self, *args, **kwargs):
        """Extend DataFrame._build_data add line current converter."""
        attrs = self.metaframe.required + list(kwargs)  # record passed attrs
        data = super()._build_data(*args, **kwargs)
        if 'It' in attrs and 'Ic' not in attrs:  # patch line current
            data['Ic'] = \
                data['It'] / data.get('Nt', self.metaframe.default['Nt'])
        return data

    def update_frame(self):
        """Propagate subspace varables to frame."""
        if self._hasattr('subspace'):
            for col in [col for col in self.subspace if col in self]:
                self.set_frame(col)

    def set_frame(self, col):
        """Inflate subspace variable and setattr in frame."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(True, 'subspace'):
            value = getattr(self, col)
            if not isinstance(value, np.ndarray):
                value = value.to_numpy()
        with self.metaframe.setlock(None):
            if hasattr(self, 'subref'):  # inflate
                value = value[self.subref]
            super().__setitem__(col, value)

    def get_frame(self, col):
        """Return inflated subspace variable."""
        self.assert_in_field(col, 'subspace')
        with self.metaframe.setlock(False, 'subspace'):
            return super().__getitem__(col)

    def assert_in_field(self, col, field):
        """Check for col in metaframe.{field}, raise error if not found."""
        try:
            self.in_field(col, field)
        except AssertionError as in_field_assert:
            raise AssertionError(
                f'\'{col}\' not specified in metaframe.subspace '
                f'{self.metaframe.subspace}') from in_field_assert

    def in_field(self, col, field):
        """Return Ture if col in metaframe.{field} and hasattr(self, field)."""
        if not isinstance(col, str):
            return False
        if self._hasattr('metaframe') and self._hasattr(field):
            if hasattr(self.attrs[field], 'columns'):
                return col in self.attrs[field].columns
        return False

    def _get_col(self, key):
        """Return column label."""
        if isinstance(key, tuple):
            col = key[-1]
        else:
            col = key
        if isinstance(col, int):
            col = self.columns[col]
        return col