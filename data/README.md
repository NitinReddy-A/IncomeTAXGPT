# Knowledge base documents

Put the PDFs you want IndianTaxGPT to answer from in this folder. Subfolders are fine, for example
`data/acts/`, `data/circulars/`, `data/guides/`. PDFs here are git-ignored.

The assistant can only answer from what is indexed, so what you put here decides what it knows.

## Good sources

All of these are published free by the Government of India:

- **Income-tax Act** text: [India Code](https://www.indiacode.nic.in) and the
  [Income Tax Department](https://incometaxindia.gov.in) site. The Income-tax Act, 2025 replaces
  the Income-tax Act, 1961 from 1 April 2026. Index the Act that applies to the years you care
  about, or both.
- **Income-tax Rules**, plus the **Finance Act** for each year's rate and slab changes.
- **CBDT circulars and notifications**, which clarify how provisions are applied.
- **Taxpayer guides and FAQs** from the Income Tax Department on filing returns, TDS, advance tax
  and refunds.

## Tips

- Use PDFs that have a text layer. Scanned images without OCR produce no text and are skipped.
- Prefer clear file names such as `income-tax-act-2025.pdf`. The file name is what users see in
  citations.
- Check what will be indexed before using Pinecone:
  `python -m indiantaxgpt.ingest --dry-run`
- After replacing documents (for example, a new Finance Act), rebuild from scratch so outdated
  passages are removed: `python -m indiantaxgpt.ingest --reset`
