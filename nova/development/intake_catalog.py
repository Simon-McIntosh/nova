from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry

mycat = Catalog.from_dict({'source1': 
                           LocalCatalogEntry('name', 'description', 'csv')})