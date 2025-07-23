LOGFILE="active_calls.log"

# Get header line once and write to logfile (overwrite)
asterisk -rx "core show channels verbose" | head -1 > "$LOGFILE"
echo "" >> "$LOGFILE"

while true; do

  # Append only the active call lines (skip header and summary lines)
  asterisk -rx "core show channels verbose" | sed '1d' |grep -vE 'active channel|active call|calls processed' >> "$LOGFILE"
  echo "" >> "$LOGFILE"
  sleep 5
done
