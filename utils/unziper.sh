for f in GaitDatasetB-silh/*.tar.gz; do
  mkdir "${f%.tar.gz}"
  tar -xzf "$f" -C "${f%.tar.gz}"
done