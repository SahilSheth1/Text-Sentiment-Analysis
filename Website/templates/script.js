const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const sentimentText = document.getElementById('sentiment-text');
const positiveScoreBar = document.getElementById('positive-score');
const negativeScoreBar = document.getElementById('negative-score');
const wordCountDisplay = document.getElementById('word-count');
const charCountDisplay = document.getElementById('char-count');
const resultSection = document.getElementById('result');

const API_ENDPOINT = 'http://localhost:8000/predict';
const MAX_CHARACTERS = 5000;

function sanitizeText(text) {
    // Remove extra whitespace and trim
    return text.replace(/\s+/g, ' ').trim();
}

function updateTextStats() {
    const rawText = textInput.value;
    const text = sanitizeText(rawText);
    
    // Count words, handling multiple spaces and trims
    const wordCount = text ? text.split(/\s+/).length : 0;
    wordCountDisplay.textContent = `${wordCount} word${wordCount !== 1 ? 's' : ''}`;

    // Trim to max characters if exceeded
    if (text.length > MAX_CHARACTERS) {
        textInput.value = text.slice(0, MAX_CHARACTERS);
    }

    const charCount = Math.min(text.length, MAX_CHARACTERS);
    charCountDisplay.textContent = `${charCount} / ${MAX_CHARACTERS} characters`;

    // Disable button if no text or text too long
    analyzeBtn.disabled = charCount === 0 || charCount > MAX_CHARACTERS;
}

textInput.addEventListener('input', updateTextStats);

// Add paste event listener to handle large pastes
textInput.addEventListener('paste', (e) => {
    // Defer the check to allow the paste to complete
    setTimeout(updateTextStats, 0);
});

async function analyzeSentiment(text) {
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: sanitizeText(text) })
        });

        if (!response.ok) {
            throw new Error('Sentiment analysis request failed');
        }

        const data = await response.json();

        let positiveScore = 0;
        let negativeScore = 0;

        if (data.sentiment === 'positive') {
            positiveScore = data.confidence * 100;
            negativeScore = 100 - positiveScore;
        } else {
            negativeScore = data.confidence * 100;
            positiveScore = 100 - negativeScore;
        }

        return {
            sentiment: data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1),
            positiveScore: Math.max(0, Math.min(positiveScore, 100)),
            negativeScore: Math.max(0, Math.min(negativeScore, 100))
        };
    } catch (error) {
        console.error('Error analyzing sentiment:', error);
        alert('Failed to analyze sentiment. Please try again.');
        return null;
    }
}

analyzeBtn.addEventListener('click', async () => {
    const text = sanitizeText(textInput.value);
    
    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    try {
        const results = await analyzeSentiment(text);

        if (results) {
            resultSection.classList.add('show');

            sentimentText.textContent = `The overall sentiment is ${results.sentiment}`;

            positiveScoreBar.textContent = `Positive: ${results.positiveScore.toFixed(1)}%`;
            positiveScoreBar.style.width = `${results.positiveScore}%`;

            negativeScoreBar.textContent = `Negative: ${results.negativeScore.toFixed(1)}%`;
            negativeScoreBar.style.width = `${results.negativeScore}%`;
        }
    } catch (error) {
        console.error(error);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Sentiment';
    }
});

textInput.addEventListener('input', () => {
    resultSection.classList.remove('show');
    positiveScoreBar.style.width = '0%';
    negativeScoreBar.style.width = '0%';
    sentimentText.textContent = 'Your sentiment will appear here';
});