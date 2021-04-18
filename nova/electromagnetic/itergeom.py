"""Build ITER coilset."""
from dataclasses import dataclass
import io
import pandas

from nova.electromagnetic.coilset import CoilSet


@dataclass
class ITERgeom(CoilSet):
    """Manage ITER geometry."""

    def __post_init__(self):
        """Insert default source."""
        super().__post_init__()
        self.metadata = {'source': 'PCR'} | self.metadata
        self.insert_coil()

    def insert_coil(self):
        """Insert poloidal field coils."""
        data = pandas.read_csv(self.coildata(), delimiter='\t',
                               skiprows=1, index_col=0,  skipinitialspace=True)
        part = ['cs' if 'CS' in name else 'pf' for name in data.index]
        columns = {col: col.split(',')[0].lower() for col in data}
        columns['R, ohm'] = 'R'
        columns['N,'] = 'nturn'
        data.rename(columns=columns, inplace=True)
        data.rename(columns={'dx': 'dl', 'dz': 'dt'}, inplace=True)
        self.coil.insert(data, part=part, turn='hex')
        self.frame.multipoint.link(['CS1U', 'CS1L'])
        self.subframe.multipoint.link(['CS1U', 'CS1L'])



    def coildata(self):
        """Return poloidal field coil geometrical data."""
        if self.metadata['source'] == 'PCR':  # update, post 2012
            return io.StringIO(
                '''
                X, m	Z, m	DX, m	DZ, m	N,	R, ohm	m, Kg
                CS3U	1.6870	5.4640	0.7400	2.093	554	0.102	9.0e3
                CS2U	1.6870	3.2780	0.7400	2.093 	554	0.113	10.0e3
                CS1U	1.6870	1.0920	0.7400	2.093 	554	0.124	11.0e3
                CS1L	1.6870	-1.0720	0.7400	2.093 	554	0.124	11.0e3
                CS2L	1.6870	-3.2580	0.7400	2.093 	554	0.113	10.0e3
                CS3L	1.6870	-5.4440	0.7400	2.093 	554	0.090	8.0e3
                PF1	3.9431	7.5641	0.9590	0.9841	248.64	0.0377	7.5e3
                PF2	8.2851	6.5298	0.5801	0.7146	115.20	0.0283	10.0e3
                PF3	11.9919	3.2652	0.6963	0.9538	185.92	0.0961	34.0e3
                PF4	11.9630	-2.2336	0.6382	0.9538	169.92	0.0791	28.0e3
                PF5	8.3908	-6.7369	0.8125	0.9538	216.80	0.0791	28.0e3
                PF6	4.3340	-7.4765	1.5590	1.1075	459.36	0.120	24.0e3
                ''')
        if self.metadata['source'] == 'baseline':  # old
            return io.StringIO(
                '''
                X, m	Z, m	DX, m	DZ, m	N,	R, ohm	m, Kg
                CS3U	1.722	5.313	0.719	2.075	554	0.102	9.0e3
                CS2U	1.722	3.188	0.719	2.075	554	0.113	10.0e3
                CS1U	1.722	1.063	0.719	2.075	554	0.124	11.0e3
                CS1L	1.722	-1.063	0.719	2.075	554	0.124	11.0e3
                CS2L	1.722	-3.188	0.719	2.075	554	0.113	10.0e3
                CS3L	1.722	-5.313	0.719	2.075	554	0.090	8.0e3
                PF1	3.9431	7.5641	0.9590	0.9841	248.64	0.0377	7.5e3
                PF2	8.2851	6.5298	0.5801	0.7146	115.20	0.0283	10.0e3
                PF3	11.9919	3.2652	0.6963	0.9538	185.92	0.0961	34.0e3
                PF4	11.9630	-2.2336	0.6382	0.9538	169.92	0.0791	28.0e3
                PF5	8.3908	-6.7369	0.8125	0.9538	216.80	0.0791	28.0e3
                PF6	4.3340	-7.4765	1.5590	1.1075	459.36	0.120	24.0e3
                ''')
        raise IndexError(f'source {self.metadata["source"]} not found')


if __name__ == '__main__':

    coilset = ITERgeom(dcoil=0.5, dplasma=0.5)
    coilset.plasma.insert({'elp': [6.5, 0, 5, 8]})
    coilset.plot()
