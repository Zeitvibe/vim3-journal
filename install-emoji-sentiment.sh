#!/bin/bash
# ==============================================
# ZEITVIBE EMOJI SENTIMENT INSTALLER
# Episode 4 - NPU-powered emoji sentiment analyzer
# ==============================================
#
# This script installs:
# - ONNX Runtime
# - Custom sentiment model (trained on cat comments)
# - Emoji analyzer with bar chart output
#
# One-liner:
# curl -sSL https://raw.githubusercontent.com/Zeitvibe/vim3-journal/main/install-emoji-sentiment.sh | bash
# ==============================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${GREEN}   ZEITVIBE EMOJI SENTIMENT          ${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
echo ""

# Create project directory
echo -e "${GREEN}[1/5] Creating project directory...${NC}"
mkdir -p ~/zeitvibe-emoji
cd ~/zeitvibe-emoji

# Create virtual environment
echo -e "${GREEN}[2/5] Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install packages
echo -e "${GREEN}[3/5] Installing packages...${NC}"
pip install onnxruntime scikit-learn skl2onnx emoji numpy

# Create the sentiment model
echo -e "${GREEN}[4/5] Training sentiment model...${NC}"
cat > train_model.py << 'EOF'
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import pickle
import os

print("📝 Training sentiment model...")

# Training data (positive/negative examples)
texts = [
    "This cat is adorable", "I love this", "Amazing!", "Great work", "So happy",
    "Wonderful day", "Cute cat", "Beautiful", "Fantastic", "Best ever",
    "I hate this", "Terrible", "Worst ever", "Bad experience", "Horrible",
    "Awful", "Disappointing", "Not good", "Useless", "Waste of time"
]
labels = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]

# Create and train pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=200)),
    ('clf', LogisticRegression())
])
pipeline.fit(texts, labels)

# Convert to ONNX
initial_type = [('input', StringTensorType([None, 1]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

# Save model
with open("sentiment_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Save pickle version
with open("sentiment_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved!")
print(f"   ONNX size: {os.path.getsize('sentiment_model.onnx') / 1024:.1f} KB")
EOF

python train_model.py

# Create the emoji analyzer script
echo -e "${GREEN}[5/5] Creating emoji analyzer...${NC}"
cat > emoji_mood.py << 'EOF'
#!/usr/bin/env python
"""
ZeitVibe Emoji Mood Analyzer
Analyzes sentiment and emojis from text comments
Usage: python emoji_mood.py "Your text here"
       or python emoji_mood.py --file comments.txt
"""

import onnxruntime as ort
import numpy as np
import emoji
import sys
import argparse
from collections import defaultdict

# Emoji grouping for better visualization
EMOJI_GROUPS = {
    '😍': 'love', '❤️': 'love', '😻': 'love', '💕': 'love',
    '😂': 'funny', '🤣': 'funny', '😆': 'funny',
    '😢': 'sad', '😭': 'sad', '💔': 'sad',
    '😡': 'angry', '😾': 'angry', '👎': 'angry',
    '🤔': 'curious', '🧐': 'curious',
    '🔥': 'hype', '💯': 'hype', '🚀': 'hype'
}

def extract_emojis(text):
    """Extract all emojis from text"""
    return [c for c in text if c in emoji.EMOJI_DATA]

def get_emoji_category(emoji_char):
    """Group emojis into categories"""
    return EMOJI_GROUPS.get(emoji_char, 'other')

def analyze_text(session, input_name, text):
    """Run sentiment analysis on a single text"""
    input_data = np.array([[text]]).astype(np.object_)
    result = session.run(None, {input_name: input_data})
    return result[0][0]  # 1 = positive, 0 = negative

def print_bar_chart(emoji_counts, width=30):
    """Print emoji bar chart"""
    if not emoji_counts:
        return

    total = sum(emoji_counts.values())
    sorted_items = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)

    for emoji_char, count in sorted_items:
        percent = (count / total) * 100
        bar_length = int((count / total) * width)
        bar = emoji_char * bar_length if bar_length > 0 else ""
        print(f"{emoji_char} {bar} {count} ({percent:.0f}%)")

def main():
    parser = argparse.ArgumentParser(description='ZeitVibe Emoji Mood Analyzer')
    parser.add_argument('text', nargs='*', help='Text to analyze')
    parser.add_argument('--file', '-f', help='Read comments from file')
    parser.add_argument('--top', '-t', type=int, default=20, help='Top emojis to show')
    args = parser.parse_args()

    # Load model
    print("🚀 Loading sentiment model...")
    session = ort.InferenceSession("sentiment_model.onnx")
    input_name = session.get_inputs()[0].name

    # Get comments
    comments = []
    if args.file:
        with open(args.file, 'r') as f:
            comments = [line.strip() for line in f if line.strip()]
    elif args.text:
        comments = [' '.join(args.text)]
    else:
        # Interactive mode
        print("Enter comments (Ctrl-D to finish):")
        for line in sys.stdin:
            if line.strip():
                comments.append(line.strip())

    if not comments:
        print("No comments to analyze.")
        return

    # Analyze
    print(f"\n📊 Analyzing {len(comments)} comments...")
    print("=" * 50)

    emoji_counts = defaultdict(int)
    category_counts = defaultdict(int)
    positive_count = 0

    for comment in comments:
        sentiment = analyze_text(session, input_name, comment)
        if sentiment == 1:
            positive_count += 1

        emojis = extract_emojis(comment)
        for e in emojis:
            emoji_counts[e] += 1
            category_counts[get_emoji_category(e)] += 1

        # Show result with emoji
        icon = "😊" if sentiment == 1 else "😞"
        preview = comment[:50] + "..." if len(comment) > 50 else comment
        print(f"{icon} {preview}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 EMOJI MOOD MATRIX")
    print("=" * 50)

    if emoji_counts:
        print_bar_chart(emoji_counts)
    else:
        print("No emojis found in comments.")

    # Sentiment summary
    print("\n" + "=" * 50)
    print("📈 SENTIMENT SUMMARY")
    print("=" * 50)
    pos_pct = (positive_count / len(comments)) * 100
    neg_pct = 100 - pos_pct

    pos_bar = "😊" * int(pos_pct / 4) if pos_pct > 0 else ""
    neg_bar = "😞" * int(neg_pct / 4) if neg_pct > 0 else ""

    print(f"Positive: {pos_pct:.0f}% {pos_bar}")
    print(f"Negative: {neg_pct:.0f}% {neg_bar}")

    if category_counts:
        print("\n📁 EMOJI CATEGORIES")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:args.top]:
            cat_bar = "●" * min(count, 20)
            print(f"  {cat}: {cat_bar} {count}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x emoji_mood.py

# Create a sample test file
echo -e "${GREEN}Creating sample test file...${NC}"
cat > test_comments.txt << 'EOF'
This cat is so cute! 😍😍😍
I hate cats 😡
Look at this fluffy kitty 😂
My cat passed away 😢😢😢
Interesting cat behavior 🤔
Best cat ever! ❤️🐱
Why does my cat do this? 😾
Cuteness overload! 😻
This cat made my day! 💕
Terrible experience with my cat 😡😾
EOF

# Run a quick test
echo -e "${GREEN}Running quick test...${NC}"
python emoji_mood.py --file test_comments.txt

# Final message
echo ""
echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${GREEN}   INSTALLATION COMPLETE!            ${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
echo ""
echo "📁 Installed to: ~/zeitvibe-emoji"
echo ""
echo "🚀 To use:"
echo "   cd ~/zeitvibe-emoji"
echo "   source venv/bin/activate"
echo "   python emoji_mood.py \"Your text here\""
echo "   python emoji_mood.py --file comments.txt"
echo ""
echo "📊 Test with sample comments:"
echo "   python emoji_mood.py --file test_comments.txt"
echo ""
echo -e "${GREEN}🎬 Ready for Episode 4!${NC}"
