You classify document chunks by ROLE for a retrieval system. For each chunk choose ONE:
- content: answer-bearing — EXPLAINS/STATES facts, gives specifications, values, steps, definitions, OR holds DATA. A data table (values, register fields, measurements, status rows — even sparse or heavily delimited with | or whitespace) is CONTENT.
- navigation: a TABLE OF CONTENTS / INDEX (section titles mapped to page numbers, dotted leaders), a list of bare section headings, or a cross-reference pointer ('see chapter X'). Its fingerprint is section-names+page-numbers, NOT data values.
- boilerplate: title page, copyright / proprietary / legal / trademark notice, document-metadata front-matter.
Judge by FUNCTION. When unsure between content and navigation, choose CONTENT — never drop real data. Choose navigation ONLY when it is clearly a ToC/index/pointer with no data.

Chunks:
{{ chunks }}

Return ONLY JSON: {"roles":[{"i":0,"role":"content|navigation|boilerplate"}, ...]} with EXACTLY one entry per chunk index.
