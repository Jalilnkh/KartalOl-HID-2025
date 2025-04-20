for d in GaitDatasetB-silh/*; do
  mv "$d"/* "$(dirname "$d")"/
  rmdir "$d"
done
