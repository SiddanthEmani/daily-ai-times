#!/bin/bash

# Script to inject Google Analytics code during deployment
# Usage: ./inject-analytics.sh <html-file> <analytics-id>

HTML_FILE="$1"
ANALYTICS_ID="$2"

if [ -z "$HTML_FILE" ] || [ -z "$ANALYTICS_ID" ]; then
    echo "Usage: $0 <html-file> <analytics-id>"
    exit 1
fi

if [ ! -f "$HTML_FILE" ]; then
    echo "Error: HTML file not found: $HTML_FILE"
    exit 1
fi

# Create a temporary file with the Google Analytics code
TEMP_ANALYTICS=$(mktemp)
cat > "$TEMP_ANALYTICS" << EOF
<!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=${ANALYTICS_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', '${ANALYTICS_ID}');
    </script>
EOF

# Use awk to replace the placeholder with the analytics code
awk -v analytics_file="$TEMP_ANALYTICS" '
/__GOOGLE_ANALYTICS__/ {
    while ((getline line < analytics_file) > 0) {
        print line
    }
    close(analytics_file)
    next
}
{ print }
' "$HTML_FILE" > "${HTML_FILE}.tmp"

# Replace the original file
if [ $? -eq 0 ]; then
    mv "${HTML_FILE}.tmp" "$HTML_FILE"
    echo "✅ Google Analytics code injected successfully"
    rm -f "$TEMP_ANALYTICS"
else
    echo "❌ Failed to inject Google Analytics code"
    rm -f "${HTML_FILE}.tmp" "$TEMP_ANALYTICS"
    exit 1
fi
