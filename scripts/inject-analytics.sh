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

# Create the Google Analytics code
ANALYTICS_CODE='<!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id='$ANALYTICS_ID'"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('"'"'js'"'"', new Date());

      gtag('"'"'config'"'"', '"'"''$ANALYTICS_ID''"'"');
    </script>'

# Replace the placeholder with the actual code
sed -i.bak "s|__GOOGLE_ANALYTICS__|$ANALYTICS_CODE|g" "$HTML_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Google Analytics code injected successfully"
    rm -f "$HTML_FILE.bak"
else
    echo "❌ Failed to inject Google Analytics code"
    exit 1
fi
