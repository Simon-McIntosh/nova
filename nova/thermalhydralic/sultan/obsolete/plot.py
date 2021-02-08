

class SultanPlot:
    """Plot methods mixin."""

    def _get_marker(self, steady=True, location='max'):
        marker = {'ls': 'none', 'alpha': 1, 'ms': 4, 'mew': 1}
        if location == 'eoh':
            marker.update({'color': 'C6', 'label': 'eoh', 'marker': 'o'})
        elif location == 'max':
            marker.update({'color': 'C4', 'label': 'max', 'marker': 'd'})
        else:
            raise IndexError(f'location {location} not in [eof, max]')
        if not steady:
            marker.update({'mfc': 'w'})
        return marker
