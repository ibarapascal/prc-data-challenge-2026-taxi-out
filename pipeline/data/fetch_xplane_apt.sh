#!/usr/bin/env bash
# fetch_xplane_apt.sh - download the apt.dat airport layouts of the ten airports from the X-Plane Scenery Gateway API
# (https://gateway.x-plane.com/apiv1); input of stand_merge.py and build_tabular_features.py.
#
# Usage: bash fetch_xplane_apt.sh <output dir>        e.g. bash fetch_xplane_apt.sh "$PRC_ROOT/final/ext/xplane-apt"
#   Default: fetch the scenery packs we used, by their Gateway scenery IDs (pinned_id below). These were the recommended
#   packs of the ten airports when we downloaded them on 2026-09-25. SHA-256 of the apt.dat we used:
#     EGLL 111341 63bde47e67dc3771115e2cd72330d169cc51c64a1d0df99d84c3ead7cf8fad82
#     LEMD  99851 4c5eb8cb712a3fd80ed9f4c9452120c1afe6feed55ad0c01f1632066184e03ff
#     LIRF 110482 34ec83782108ec317f240c679d6f91a1ec0920751495ec140b694e129a91c59c
#     LTFM 105504 cb849f454addd9a611c3d96ba46b8138b8da12eb80120c0efdcfa0369c54ba39
#     LSZH 106102 d530c122ac93bda5746a9868e6e6bcbd8a3949dfd004834f1fe229a3f1acff98
#     LFPG 111022 b1787f8cbd8378f693b6726d6649014a01bcc5c0a329f21898b11d74b10f7090
#     LEBL 102255 700163e6f017c5fea72afe0cb6e3213e7f7a25060c26eab4d183f14538e9809b
#     EDDF 103368 065c3fb5de426519df0afba3df27af9429871c8174ea3d04cffa6c27cba71bea
#     EHAM 111943 a10e69d92923067af7d5f7c378c62f48bc404b468b3c8cc8b90a17a30d41f309
#     EDDM 111124 d7fa09fb2c0cba1d59101d545203018eb8e475a5621ca82b6a02ba497010b6e4
#   RECOMMENDED=1: fetch the pack that the Gateway currently recommends for each airport instead.
# Output per airport: <ICAO>/airport.json, <ICAO>/scenery_meta.json (scenery metadata without the zip blob),
#   <ICAO>/scenery_<id>.zip, <ICAO>/unzip_<id>/, <ICAO>/apt.dat (the .dat file of the pack); plus manifest.csv
#   (ICAO, scenery ID, author, acceptance and approval dates, SHA-256 of apt.dat, fetch time).
# Requires curl, jq, unzip, shasum and base64 (-D is the macOS decode flag; GNU coreutils use -d).
# The downloaded data are not part of this repository; see ATTRIBUTION.md for their terms.
set -uo pipefail
OUT=${1:?usage: fetch_xplane_apt.sh <output dir>}
mkdir -p "$OUT" && cd "$OUT" || exit 1
API=https://gateway.x-plane.com/apiv1
C="curl -sS --fail --max-time 120 --limit-rate 1000000"
pinned_id() {
  case $1 in
    EGLL) echo 111341 ;; LEMD) echo 99851 ;; LIRF) echo 110482 ;; LTFM) echo 105504 ;; LSZH) echo 106102 ;;
    LFPG) echo 111022 ;; LEBL) echo 102255 ;; EDDF) echo 103368 ;; EHAM) echo 111943 ;; EDDM) echo 111124 ;;
  esac
}
echo "icao,sceneryId,userName,dateAccepted,dateApproved,apt_sha256,fetched_utc" > manifest.csv
for a in EGLL LEMD LIRF LTFM LSZH LFPG LEBL EDDF EHAM EDDM; do
  mkdir -p "$a"
  $C "$API/airport/$a" -o "$a/airport.json" || { echo "FAIL airport $a"; continue; }
  if [ "${RECOMMENDED:-0}" = 1 ]; then id=$(jq -r '.airport.recommendedSceneryId' "$a/airport.json"); else id=$(pinned_id "$a"); fi
  $C "$API/scenery/$id" -o "$a/scenery_meta_full.json" || { echo "FAIL scenery $a $id"; continue; }
  jq -r '.scenery.masterZipBlob' "$a/scenery_meta_full.json" | base64 -D > "$a/scenery_$id.zip"
  jq 'del(.scenery.masterZipBlob)' "$a/scenery_meta_full.json" > "$a/scenery_meta.json" && rm "$a/scenery_meta_full.json"
  (cd "$a" && unzip -o -q "scenery_$id.zip" -d "unzip_$id")
  f=$(find "$a/unzip_$id" -iname '*.dat' | head -1)
  [ -n "$f" ] && cp "$f" "$a/apt.dat"
  sha=$(shasum -a 256 "$a/apt.dat" 2>/dev/null | cut -d' ' -f1)
  meta=$(jq -r '[.scenery.userName, .scenery.dateAccepted, .scenery.dateApproved] | @csv' "$a/scenery_meta.json")
  echo "$a,$id,$meta,$sha,$(date -u +%FT%TZ)" >> manifest.csv
  echo "OK $a $id $(wc -l < "$a/apt.dat") lines"
  sleep 2
done
echo DONE
