document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'http://localhost:8000/recommend';

    const elements = {
        form: document.getElementById('recommendation-form'),
        userInput: document.getElementById('user-input'),
        resultsContainer: document.getElementById('results-container'),
        errorMessage: document.getElementById('error-message'),
    };
    
    const showSkeletonLoader = (count = 5) => {
        elements.resultsContainer.innerHTML = '';
        for (let i = 0; i < count; i++) {
            const skeletonCard = `
                <div class="skeleton-card">
                    <div class="skeleton-img"></div>
                    <div class="skeleton-text">
                        <div class="skeleton-line h3"></div>
                        <div class="skeleton-line p-short"></div>
                        <div class="skeleton-line p"></div>
                    </div>
                </div>`;
            elements.resultsContainer.innerHTML += skeletonCard;
        }
    };

    const displayResults = (recommendations) => {
        elements.resultsContainer.innerHTML = '';
        if (recommendations.length === 0) {
            displayError("No movies found matching your criteria.");
            return;
        }
        recommendations.forEach(film => {
            const imageHtml = film.poster_url 
                ? `<img src="${film.poster_url}" alt="${film.title} poster" onerror="this.remove();">`
                : '';

            const filmCard = `
                <div class="film-card">
                    ${imageHtml} 
                    <div class="film-card-content">
                        <h3>${film.title}</h3>
                        <div class="tur"><strong>Genre:</strong> ${film.genre || 'N/A'}</div>
                        <p>${film.overview}</p>
                    </div>
                </div>`;
            elements.resultsContainer.innerHTML += filmCard;
        });
    };
    
    const displayError = (message) => {
        elements.resultsContainer.innerHTML = '';
        elements.errorMessage.textContent = `Error: ${message}`;
        elements.errorMessage.classList.remove('hidden');
    };

    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = elements.userInput.value.trim();
        if (!text) return;
        
        elements.errorMessage.classList.add('hidden');
        showSkeletonLoader();

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, top_k: 10 })
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "An unknown error occurred.");
            }
            const data = await response.json();
            displayResults(data.recommendations);
        } catch (error) {
            displayError(error.message);
        }
    });
});
