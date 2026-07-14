// Vocabulario Inicial (Base de Datos Local)
let vocabulary = [
    {
        word: "reluctant",
        category: "Adjetivo",
        pronunciation: "/rɪˈlʌk.tənt/",
        translation: "renuente / inclinado a evitar",
        example_en: "She was reluctant to admit she was wrong.",
        example_es: "Ella era renuente a admitir que estaba equivocada.",
        difficulty: "Media",
        reviews: 3,
        accuracy: 0.85,
        elapsedHours: 72.0, // 3 días
        latency: 1800,
        intervalDaysPrev: 3
    },
    {
        word: "unprecedented",
        category: "Adjetivo",
        pronunciation: "/ʌnˈpres.ɪ.den.tɪd/",
        translation: "sin precedentes",
        example_en: "The team has enjoyed unprecedented success this year.",
        example_es: "El equipo ha disfrutado de un éxito sin precedentes este año.",
        difficulty: "Alta",
        reviews: 1,
        accuracy: 0.50,
        elapsedHours: 120.0, // 5 días
        latency: 4500,
        intervalDaysPrev: 1
    },
    {
        word: "mitigate",
        category: "Verbo Regular",
        pronunciation: "/ˈmɪt.ɪ.ɡeɪt/",
        translation: "mitigar / atenuar",
        example_en: "Soil erosion was mitigated by planting trees.",
        example_es: "La erosión del suelo se mitigó plantando árboles.",
        difficulty: "Media",
        reviews: 4,
        accuracy: 0.90,
        elapsedHours: 48.0, // 2 días
        latency: 1200,
        intervalDaysPrev: 4
    },
    {
        word: "compelling",
        category: "Adjetivo",
        pronunciation: "/kəmˈpel.ɪŋ/",
        translation: "convincente / atractivo",
        example_en: "The lawyer presented a compelling argument.",
        example_es: "El abogado presentó un argumento convincente.",
        difficulty: "Media",
        reviews: 2,
        accuracy: 0.80,
        elapsedHours: 96.0, // 4 días
        latency: 2200,
        intervalDaysPrev: 2
    },
    {
        word: "cognitive",
        category: "Adjetivo",
        pronunciation: "/ˈkɒɡ.nɪ.tɪv/",
        translation: "cognitivo",
        example_en: "Cognitive development is fast in early childhood.",
        example_es: "El desarrollo cognitivo es rápido en la infancia temprana.",
        difficulty: "Media",
        reviews: 5,
        accuracy: 0.95,
        elapsedHours: 168.0, // 7 días
        latency: 900,
        intervalDaysPrev: 5
    },
    {
        word: "phrasal verb",
        category: "Sustantivo",
        pronunciation: "/ˈfreɪ.zəl vɜːb/",
        translation: "verbo compuesto",
        example_en: "'Give up' is a common phrasal verb meaning to surrender.",
        example_es: "'Give up' es un verbo compuesto común que significa rendirse.",
        difficulty: "Alta",
        reviews: 2,
        accuracy: 0.60,
        elapsedHours: 24.0, // 1 día
        latency: 3800,
        intervalDaysPrev: 2
    },
    {
        word: "retention",
        category: "Sustantivo",
        pronunciation: "/rɪˈten.ʃən/",
        translation: "retención",
        example_en: "Visual aids improve the retention of vocabulary.",
        example_es: "Las ayudas visuales mejoran la retención de vocabulario.",
        difficulty: "Baja",
        reviews: 6,
        accuracy: 0.98,
        elapsedHours: 240.0, // 10 días
        latency: 700,
        intervalDaysPrev: 8
    },
    {
        word: "decay",
        category: "Verbo Regular",
        pronunciation: "/dɪˈkeɪ/",
        translation: "decaer / desvanecer",
        example_en: "Without practice, your L2 vocabulary will decay.",
        example_es: "Sin práctica, tu vocabulario de L2 decaerá.",
        difficulty: "Baja",
        reviews: 3,
        accuracy: 0.75,
        elapsedHours: 144.0, // 6 días
        latency: 1600,
        intervalDaysPrev: 3
    },
    {
        word: "retrieve",
        category: "Verbo Regular",
        pronunciation: "/rɪˈtriːv/",
        translation: "recuperar / evocar",
        example_en: "It was difficult to retrieve the word from memory.",
        example_es: "Fue difícil recuperar la palabra de la memoria.",
        difficulty: "Media",
        reviews: 4,
        accuracy: 0.85,
        elapsedHours: 12.0, // 0.5 días
        latency: 1100,
        intervalDaysPrev: 4
    },
    {
        word: "spaced repetition",
        category: "Sustantivo",
        pronunciation: "/speɪst ˌrep.ɪˈtɪʃ.ən/",
        translation: "repetición espaciada",
        example_en: "Spaced repetition helps you review words at the right time.",
        example_es: "La repetición espaciada te ayuda a revisar palabras en el momento adecuado.",
        difficulty: "Alta",
        reviews: 3,
        accuracy: 0.70,
        elapsedHours: 180.0, // 7.5 días
        latency: 3200,
        intervalDaysPrev: 3
    }
];

// Clonar datos iniciales para reseteos
const initialVocabulary = JSON.parse(JSON.stringify(vocabulary));

// Variables de Estado de la App
let currentWordIndex = 0;
let userMcer = "B1";
let cardLoadStartTime = 0;
let answerLogged = false;

// Configuración del Círculo de Carga del Simulador
const circle = document.getElementById('sim-progress-circle');
let radius, circumference;
if (circle) {
    radius = circle.r.baseVal.value;
    circumference = radius * 2 * Math.PI;
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = circumference;
}

// ----------------- MOTOR DE INFERENCIA DE MACHINE LEARNING (Regresión Logística Local) -----------------
function predecirRiesgoOlvido(horas, repasos, aciertos, dificultad, latencia, mcer) {
    // Mapeo ordinal de dificultad
    const diffMap = { 'Baja': 1, 'Media': 2, 'Alta': 3 };
    const diffNum = diffMap[dificultad] || 2;

    // Mapeo ordinal de nivel de competencia
    const mcerMap = { 'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6 };
    const mcerNum = mcerMap[mcer] || 3;

    // Ecuación de Regresión Logística basada en los coeficientes aprendidos en Python
    let logit = (0.022 * horas)
              - (0.35 * repasos)
              - (3.5 * aciertos)
              + (0.9 * diffNum)
              + (0.0004 * latencia)
              - (0.5 * mcerNum)
              + 0.8;

    let probOlvido = 1 / (1 + Math.exp(-logit));
    let necesitaRepaso = probOlvido > 0.5 ? 1 : 0;

    // Lógica del Algoritmo de Repetición Espaciada
    let diasEspera = 1;
    if (necesitaRepaso === 0) {
        let intervaloBase = 2.5 * repasos;
        // El intervalo sugerido se expande con menores riesgos de olvido
        diasEspera = Math.round(Math.min(Math.max(intervaloBase * (1.5 - probOlvido), 2), 90));
    }

    return {
        probabilidad: probOlvido,
        necesitaRepaso: necesitaRepaso,
        diasEspera: diasEspera
    };
}

// ----------------- CONTROLADORES DE PANTALLAS (TABS) -----------------
const navItems = document.querySelectorAll('.nav-item');
const tabPanels = document.querySelectorAll('.tab-panel');
const pageTitle = document.getElementById('page-title');
const pageSubtitle = document.getElementById('page-subtitle');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        // Remover clases activas
        navItems.forEach(nav => nav.classList.remove('active'));
        tabPanels.forEach(panel => panel.classList.remove('active'));

        // Agregar activa al botón
        item.classList.add('active');

        // Mostrar panel correspondiente
        const tabName = item.getAttribute('data-tab');
        const targetPanel = document.getElementById(`tab-${tabName}`);
        if (targetPanel) targetPanel.classList.add('active');

        // Actualizar textos de header
        if (tabName === 'practice') {
            pageTitle.innerText = "Práctica Activa de Vocabulario";
            pageSubtitle.innerText = "Refuerza tu memoria en el momento exacto usando Machine Learning.";
            resetPracticeCard();
        } else if (tabName === 'simulator') {
            pageTitle.innerText = "Simulador Paramétrico del Modelo ML";
            pageSubtitle.innerText = "Interactúa directamente con la Regresión Logística variando los coeficientes de telemetría.";
            runSimulator();
        } else if (tabName === 'dashboard') {
            pageTitle.innerText = "Panel de Monitoreo de Retención";
            pageSubtitle.innerText = "Estadísticas del vocabulario y cronograma de repetición espaciada.";
            renderDashboard();
        }
    });
});

// ----------------- TAB 1: PRÁCTICA ACTIVA (FLASHCARDS) -----------------
const flashcard = document.getElementById('flashcard');
const cardCategory = document.getElementById('card-category');
const cardWord = document.getElementById('card-word');
const cardPronunciation = document.getElementById('card-pronunciation');
const cardTranslation = document.getElementById('card-translation');
const cardExampleEn = document.getElementById('card-example-en');
const cardExampleEs = document.getElementById('card-example-es');

const btnForgot = document.getElementById('btn-forgot');
const btnRemembered = document.getElementById('btn-remembered');
const btnNextCard = document.getElementById('btn-next-card');
const cardProgressEl = document.getElementById('card-progress');

// Cola de práctica (orden de palabras para la sesión)
let practiceQueue = [];
let practiceQueueIndex = 0;

const latencyText = document.getElementById('telemetry-latency');
const difficultyText = document.getElementById('telemetry-difficulty');
const reviewsText = document.getElementById('telemetry-reviews');
const accuracyText = document.getElementById('telemetry-accuracy');
const elapsedText = document.getElementById('telemetry-elapsed');

const practiceProbVal = document.getElementById('practice-prob-val');
const practiceProbFill = document.getElementById('practice-prob-fill');
const practiceRecBadge = document.getElementById('practice-rec-badge');

// Voltear Tarjeta
if (flashcard) {
    flashcard.addEventListener('click', () => {
        flashcard.classList.toggle('flipped');
    });
}

function buildPracticeQueue() {
    // Ordenar todas las palabras por riesgo de olvido (mayor primero)
    practiceQueue = vocabulary.map((word, idx) => {
        let prediction = predecirRiesgoOlvido(
            word.elapsedHours,
            word.reviews,
            word.accuracy,
            word.difficulty,
            word.latency,
            userMcer
        );
        return { index: idx, prob: prediction.probability };
    }).sort((a, b) => b.prob - a.prob);
    practiceQueueIndex = 0;
}

function updateProgressCounter() {
    if (cardProgressEl) {
        cardProgressEl.innerText = `Tarjeta ${practiceQueueIndex + 1} de ${practiceQueue.length}`;
    }
}

function resetPracticeCard() {
    if (vocabulary.length === 0) return;
    buildPracticeQueue();
    currentWordIndex = practiceQueue[0].index;
    loadWordToCard(vocabulary[currentWordIndex]);
    updateProgressCounter();
    // Asegurar botón siguiente activo
    if (btnNextCard) {
        btnNextCard.disabled = false;
        btnNextCard.style.opacity = '1';
    }
}

function goToNextCard() {
    practiceQueueIndex = (practiceQueueIndex + 1) % practiceQueue.length;
    currentWordIndex = practiceQueue[practiceQueueIndex].index;
    loadWordToCard(vocabulary[currentWordIndex]);
    updateProgressCounter();
    // Restablecer botones de respuesta
    btnForgot.disabled = false;
    btnRemembered.disabled = false;
    btnForgot.style.opacity = '1';
    btnRemembered.style.opacity = '1';
}

function loadWordToCard(item) {
    if (!item) return;

    // Cargar contenido
    cardWord.innerText = item.word;
    cardCategory.innerText = item.category;
    cardPronunciation.innerText = item.pronunciation;
    cardTranslation.innerText = item.translation;
    cardExampleEn.innerText = item.example_en;
    cardExampleEs.innerText = item.example_es;

    // Resetear rotación de tarjeta
    flashcard.classList.remove('flipped');

    // Registrar inicio del tiempo de respuesta
    cardLoadStartTime = performance.now();
    answerLogged = false;

    // Desbloquear botones de respuesta
    btnForgot.disabled = false;
    btnRemembered.disabled = false;
    btnForgot.style.opacity = "1";
    btnRemembered.style.opacity = "1";

    // Mostrar métricas actuales de telemetría previa
    latencyText.innerText = "Mediendo...";
    difficultyText.innerText = item.difficulty;
    reviewsText.innerText = item.reviews;
    accuracyText.innerText = `${Math.round(item.accuracy * 100)}%`;
    
    let days = Math.round(item.elapsedHours / 24);
    elapsedText.innerText = `${Math.round(item.elapsedHours)}h (${days === 1 ? '1 día' : days + ' días'})`;

    // Predicción actual del modelo antes del repaso
    updatePredictionDisplay(item);
}

function updatePredictionDisplay(item) {
    let pred = predecirRiesgoOlvido(
        item.elapsedHours,
        item.reviews,
        item.accuracy,
        item.difficulty,
        item.latency,
        userMcer
    );

    let percent = Math.round(pred.probability * 100);
    practiceProbVal.innerText = `${percent}%`;
    practiceProbFill.style.width = `${percent}%`;

    // Cambiar color de barra según probabilidad
    if (pred.probability > 0.5) {
        practiceProbFill.style.background = "linear-gradient(90deg, #f87171, #ef4444)";
        practiceRecBadge.innerText = "¡Riesgo Alto de Olvido! - Repasar Hoy";
        practiceRecBadge.className = "recommendation-badge repaso";
    } else {
        practiceProbFill.style.background = "linear-gradient(90deg, #34d399, #10b981)";
        practiceRecBadge.innerText = `Retención Estable - Próximo repaso sugerido en ${pred.diasEspera} días`;
        practiceRecBadge.className = "recommendation-badge estable";
    }
}

function logStudentAnswer(wasCorrect) {
    if (answerLogged) return;
    answerLogged = true;

    // Bloquear botones
    btnForgot.disabled = true;
    btnRemembered.disabled = true;
    btnForgot.style.opacity = "0.5";
    btnRemembered.style.opacity = "0.5";

    // Calcular latencia exacta en milisegundos
    let endTime = performance.now();
    let currentLatency = Math.round(endTime - cardLoadStartTime);
    latencyText.innerText = `${currentLatency} ms`;

    // Obtener la palabra actual
    let item = vocabulary[currentWordIndex];

    // Actualizar sus estadísticas locales usando ETL lógico
    item.reviews += 1;
    // Tasa de aciertos adaptativa acumulativa
    item.accuracy = (item.accuracy * (item.reviews - 1) + (wasCorrect ? 1 : 0)) / item.reviews;
    item.latency = currentLatency;
    item.elapsedHours = 0.5; // Reseteado: acaba de repasarse ahora mismo (0.5 horas de delay simulado)

    // Mostrar feedback visual e inferencia actualizada
    updatePredictionDisplay(item);

    // Activar el botón siguiente con animación de pulso
    if (btnNextCard) {
        btnNextCard.disabled = false;
        btnNextCard.style.opacity = '1';
        btnNextCard.classList.add('pulse-ready');
    }
}

if (btnForgot) {
    btnForgot.addEventListener('click', () => logStudentAnswer(false));
}
if (btnRemembered) {
    btnRemembered.addEventListener('click', () => logStudentAnswer(true));
}
if (btnNextCard) {
    btnNextCard.addEventListener('click', () => {
        btnNextCard.classList.remove('pulse-ready');
        goToNextCard();
    });
}

// ----------------- TAB 2: SIMULADOR DE MACHINE LEARNING -----------------
const sliderElapsed = document.getElementById('slider-elapsed');
const sliderLatency = document.getElementById('slider-latency');
const sliderReviews = document.getElementById('slider-reviews');
const sliderAccuracy = document.getElementById('slider-accuracy');
const selectDifficulty = document.getElementById('select-difficulty');
const selectMcer = document.getElementById('select-mcer');

const valElapsed = document.getElementById('val-elapsed');
const valLatency = document.getElementById('val-latency');
const valReviews = document.getElementById('val-reviews');
const valAccuracy = document.getElementById('val-accuracy');

const simProbPercent = document.getElementById('sim-prob-percent');
const simDecisionVal = document.getElementById('sim-decision-val');
const simDecisionDesc = document.getElementById('sim-decision-desc');
const simDecisionBox = document.getElementById('sim-decision-box');
const simRecDays = document.getElementById('sim-rec-days');

function setProgressCircle(percent) {
    if (!circle) return;
    const offset = circumference - (percent / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    
    // Cambiar color según riesgo
    if (percent > 50) {
        circle.style.stroke = '#ef4444'; // Red
    } else {
        circle.style.stroke = '#10b981'; // Green
    }
}

function runSimulator() {
    let horas = parseFloat(sliderElapsed.value);
    let latencia = parseFloat(sliderLatency.value);
    let repasos = parseInt(sliderReviews.value);
    let aciertos = parseFloat(sliderAccuracy.value) / 100;
    let dificultad = selectDifficulty.value;
    let mcer = selectMcer.value;

    // Actualizar textos en los labels de los sliders
    let days = Math.round(horas / 24);
    valElapsed.innerText = `${horas} h (${days === 1 ? '1 día' : days + ' días'})`;
    valLatency.innerText = `${latencia} ms`;
    valReviews.innerText = repasos;
    valAccuracy.innerText = `${Math.round(aciertos * 100)} %`;

    // Hacer inferencia del modelo
    let result = predecirRiesgoOlvido(horas, repasos, aciertos, dificultad, latencia, mcer);
    let percent = Math.round(result.probability * 100);

    // Actualizar UI
    simProbPercent.innerText = `${percent}%`;
    setProgressCircle(percent);

    if (result.necesitaRepaso === 1) {
        simDecisionVal.innerText = "Requiere Intervención (Clase 1)";
        simDecisionVal.className = "decision-value danger";
        simDecisionDesc.innerText = "La probabilidad de olvido supera el umbral crítico tolerado (50%). Se recomienda aplicar un estímulo de refuerzo de inmediato.";
        simDecisionBox.style.borderColor = "rgba(239, 68, 68, 0.2)";
        simDecisionBox.style.backgroundColor = "rgba(239, 68, 68, 0.02)";
        simRecDays.innerText = "Hoy mismo";
        simRecDays.style.color = "#f87171";
    } else {
        simDecisionVal.innerText = "Retención Segura (Clase 0)";
        simDecisionVal.className = "decision-value success";
        simDecisionDesc.innerText = "El estudiante evoca la palabra de forma fluida y mantiene una retención estable. Se aconseja espaciar el intervalo de estudio.";
        simDecisionBox.style.borderColor = "rgba(16, 185, 129, 0.2)";
        simDecisionBox.style.backgroundColor = "rgba(16, 185, 129, 0.02)";
        simRecDays.innerText = `${result.diasEspera} días`;
        simRecDays.style.color = "#60a5fa";
    }
}

// Agregar Event Listeners a los controles del simulador
[sliderElapsed, sliderLatency, sliderReviews, sliderAccuracy, selectDifficulty, selectMcer].forEach(element => {
    if (element) {
        element.addEventListener('input', runSimulator);
        element.addEventListener('change', runSimulator);
    }
});

// ----------------- TAB 3: PANEL DE MONITOREO DE RETENCIÓN (DASHBOARD) -----------------
const dashTotalWords = document.getElementById('dash-total-words');
const dashRetainedWords = document.getElementById('dash-retained-words');
const dashForgotWords = document.getElementById('dash-forgot-words');
const vocabTableBody = document.getElementById('vocab-table-body');
const btnResetVocab = document.getElementById('btn-reset-vocab');
const userMcerSelect = document.getElementById('user-mcer-select');
const navUserLevel = document.getElementById('nav-user-level');

if (userMcerSelect) {
    userMcerSelect.addEventListener('change', (e) => {
        userMcer = e.target.value;
        const levelNames = {
            'A1': 'A1 (Principiante)',
            'A2': 'A2 (Básico)',
            'B1': 'B1 (Pre-Intermedio)',
            'B2': 'B2 (Intermedio Alto)',
            'C1': 'C1 (Avanzado)',
            'C2': 'C2 (Maestría)'
        };
        navUserLevel.innerText = `Nivel: ${levelNames[userMcer]}`;
        // Recargar si estamos en una pestaña interactiva
        resetPracticeCard();
    });
}

function renderDashboard() {
    if (!vocabTableBody) return;

    // Contadores
    let total = vocabulary.length;
    let retainedCount = 0;
    let forgotCount = 0;

    // Limpiar tabla
    vocabTableBody.innerHTML = "";

    vocabulary.forEach((word) => {
        let pred = predecirRiesgoOlvido(
            word.elapsedHours,
            word.reviews,
            word.accuracy,
            word.difficulty,
            word.latency,
            userMcer
        );

        let isAtRisk = pred.necesitaRepaso === 1;
        if (isAtRisk) {
            forgotCount++;
        } else {
            retainedCount++;
        }

        // Determinar badge de dificultad
        let diffBadge = "";
        if (word.difficulty === 'Baja') diffBadge = '<span class="badge badge-low">Baja</span>';
        else if (word.difficulty === 'Media') diffBadge = '<span class="badge badge-med">Media</span>';
        else diffBadge = '<span class="badge badge-high">Alta</span>';

        // Determinar badge de estado y fecha sugerida
        let statusBadge = "";
        if (isAtRisk) {
            statusBadge = '<span class="badge badge-danger">Repasar Hoy</span>';
        } else {
            statusBadge = `<span class="badge badge-success">Estable (${pred.diasEspera}d)</span>`;
        }

        // Traducir horas transcurridas
        let hrsStr = "";
        if (word.elapsedHours < 24) {
            hrsStr = `${Math.round(word.elapsedHours)}h`;
        } else {
            hrsStr = `${Math.round(word.elapsedHours / 24)} días`;
        }

        // Crear fila
        let row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${word.word}</strong><br><small style="color: #64748b;">${word.pronunciation}</small></td>
            <td>${word.category}</td>
            <td>${diffBadge}</td>
            <td>${word.reviews}</td>
            <td>Hace ${hrsStr}</td>
            <td>${Math.round(pred.probability * 100)}%</td>
            <td>${statusBadge}</td>
        `;
        vocabTableBody.appendChild(row);
    });

    // Actualizar tarjetas de métricas del dashboard
    dashTotalWords.innerText = total;
    dashRetainedWords.innerText = retainedCount;
    dashForgotWords.innerText = forgotCount;
}

if (btnResetVocab) {
    btnResetVocab.addEventListener('click', () => {
        vocabulary = JSON.parse(JSON.stringify(initialVocabulary));
        renderDashboard();
    });
}

// ----------------- INICIALIZACIÓN DE LA APLICACIÓN -----------------
window.addEventListener('DOMContentLoaded', () => {
    // Cargar la primera tarjeta de práctica
    resetPracticeCard();
});
