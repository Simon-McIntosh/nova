import imas

if __name__ == "__main__":

    HOST = "data.mastu.ukaea.uk"
    PORT = 56560
    MACHINE = "MASTU" #(DRAFT/MASTU)
    SHOT = 45272
    VERBOSE = 0
    BATCH_SIZE = 20
    RUN = 1
    USER = 'jg3176'

    entry = imas.DBEntry(
            f'imas://{HOST}:{PORT}/uda?'
            f'mapping={MACHINE}&path=/&'
            f'verbose={VERBOSE}&shot={SHOT}&'
            f'batch_size={BATCH_SIZE}', 'r'
    )

    ids_name = 'equilibrium'
    ids_name = 'pf_active'
    print(f'Mapping {ids_name}')
    ids = entry.get(ids_name)
    print(ids)
