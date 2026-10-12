# Data guide

This repository intentionally does **not** contain the 428 MB source archive
`Trade_DetailedTradeMatrix_E_All_Data.zip`, manuscript files, or supplementary
Word/PDF files. Obtain the exact FAOSTAT Detailed Trade Matrix archive used in
the study independently, verify its checksum, and either set `TRADE_ZIP` to its
absolute path or place it at `data/raw/Trade_DetailedTradeMatrix_E_All_Data.zip`.
The `data/raw/` directory is ignored by Git.

The archive inspected for this release contained the following seven members:

- `Trade_DetailedTradeMatrix_E_All_Data.csv`
- `Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv` (the analysis member)
- `Trade_DetailedTradeMatrix_E_Elements.csv`
- `Trade_DetailedTradeMatrix_E_Flags.csv`
- `Trade_DetailedTradeMatrix_E_ItemCodes.csv`
- `Trade_DetailedTradeMatrix_E_PartnerCountries.csv`
- `Trade_DetailedTradeMatrix_E_ReporterCountries.csv`

The expected SHA-256 of the locally archived source ZIP is
`edf36ebefc487862077deb44d421a784facb647cf38a5002a0f98ad9cbbdd90e`.

`figure_data/` contains 18 compact CSV inputs for rebuilding the supplied
independent panels. They are derived result data, not raw bilateral records.
They are included locally as **candidate release material only**. Before any
public upload, the authors must confirm publisher policy, co-author approval,
and that the intended FAOSTAT release and attribution terms permit the chosen
redistribution.

FAOSTAT source: [FAOSTAT Detailed trade matrix](https://data.fao.org/catalog/dataset/955a7b61-ce56-4a04-adf5-61a3012eec84). Consult the release-specific metadata
and [FAO Statistical Database Terms of Use](https://www.fao.org/contact-us/terms/db-terms-of-use/en) before redistributing data.
