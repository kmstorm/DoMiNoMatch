#!/bin/bash

LOG_FILE="download_log_$(date +%Y%m%d_%H%M%S).log"

domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

txt_base="https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt"

declare -A zip_links=(
  ["clipart.zip"]="https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"
  ["infograph.zip"]="https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"
  ["painting.zip"]="https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"
  ["quickdraw.zip"]="https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"
  ["real.zip"]="https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"
  ["sketch.zip"]="https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
)

# mkdir -p ~/domainnet && cd ~/domainnet || exit 1

log_status() {
  status=$1
  file=$2
  echo "[$(date +'%F %T')] [$status] $file" >> "$LOG_FILE"
}

# ZIPs
for filename in "${!zip_links[@]}"; do
  echo "Downloading $filename..."
  if wget -q -c "${zip_links[$filename]}" -O "$filename"; then
    log_status "OK" "$filename"
  else
    log_status "FAIL" "$filename"
  fi
done

# TXTs
for domain in "${domains[@]}"; do
  for split in train test; do
    file="${domain}_${split}.txt"
    url="$txt_base/$file"
    echo "Downloading $file..."
    if wget -q -c "$url" -O "$file"; then
      log_status "OK" "$file"
    else
      log_status "FAIL" "$file"
    fi
  done
done

echo "✅ Done. Summary log: $LOG_FILE"
