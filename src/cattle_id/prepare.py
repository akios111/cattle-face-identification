def main(argv: list[str] | None = None) -> None:
    print("[prepare-entry] importing cattle_id.data", flush=True)
    from .data import main as data_main

    print("[prepare-entry] running cattle_id.data.main", flush=True)
    data_main(argv)


if __name__ == "__main__":
    main()
