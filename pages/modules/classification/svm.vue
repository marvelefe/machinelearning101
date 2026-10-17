<template>
    <ModuleLayout title="Support Vector Machines (SVM)"
        description="Learn how SVM finds the maximum margin hyperplane and uses kernel functions for non-linear classification"
        difficulty="advanced" :estimated-time="35" :table-of-contents="tableOfContents"
        module-id="svm"
        :parameters="parameters"
        @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
        <!-- Content Section -->
        <template #content>
            <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
        </template>

        <!-- Interactive Section -->
        <template #interactive>
            <InteractiveSection :loading="isTraining" :progress="trainingProgress"
                loading-message="Training SVM model..." :error="error" :on-retry="handleRetry">
                <!-- Controls -->
                <template #controls>
                    <ControlPanel :controls="controls" :values="parameters" :loading="isTraining"
                        submit-label="Train Model" @change="handleParameterChange" @reset="handleReset"
                        @submit="handleTrain" />

                    <!-- Metrics Display -->
                    <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Performance</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <div class="text-xs text-gray-500">Accuracy</div>
                                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.accuracy || 0) *
                                    100).toFixed(2) }}%</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-500">Precision</div>
                                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.precision || 0) *
                                    100).toFixed(2) }}%
                                </div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-500">Recall</div>
                                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.recall || 0) *
                                    100).toFixed(2) }}%</div>
                            </div>
                            <div>
                                <div class="text-xs text-gray-500">F1 Score</div>
                                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.f1Score || 0) *
                                    100).toFixed(2) }}%</div>
                            </div>
                        </div>

                        <!-- Support Vectors Count -->
                        <div v-if="metrics.numSupportVectors !== undefined"
                            class="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
                            <div class="text-xs text-blue-700 font-medium">Support Vectors</div>
                            <div class="text-2xl font-bold text-blue-900">{{ metrics.numSupportVectors }}</div>
                            <div class="text-xs text-blue-600 mt-1">
                                {{ ((metrics.numSupportVectors / (parameters.dataset === 'binary-classification' ? 200 :
                                    100)) * 100).toFixed(1) }}% of training data
                            </div>
                        </div>

                        <!-- Confusion Matrix -->
                        <div v-if="metrics.confusionMatrix" class="mt-4">
                            <h4 class="text-xs font-semibold text-gray-700 mb-2">Confusion Matrix</h4>
                            <div class="grid grid-cols-2 gap-2 text-center text-sm">
                                <div class="bg-blue-100 p-2 rounded">
                                    <div class="text-xs text-gray-600">True Neg</div>
                                    <div class="font-semibold">{{ metrics.confusionMatrix[0][0] }}</div>
                                </div>
                                <div class="bg-red-100 p-2 rounded">
                                    <div class="text-xs text-gray-600">False Pos</div>
                                    <div class="font-semibold">{{ metrics.confusionMatrix[0][1] }}</div>
                                </div>
                                <div class="bg-red-100 p-2 rounded">
                                    <div class="text-xs text-gray-600">False Neg</div>
                                    <div class="font-semibold">{{ metrics.confusionMatrix[1][0] }}</div>
                                </div>
                                <div class="bg-green-100 p-2 rounded">
                                    <div class="text-xs text-gray-600">True Pos</div>
                                    <div class="font-semibold">{{ metrics.confusionMatrix[1][1] }}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </template>

                <!-- Visualization -->
                <template #visualization>
                    <div class="space-y-4">
                        <div class="relative">
                            <Scatter2D v-if="scatterSeries && scatterSeries.length > 0" :series="scatterSeries"
                                title="Decision Boundary & Support Vectors" x-label="Feature 1" y-label="Feature 2"
                                :show-legend="true" height="500px" />

                            <!-- Overlay heatmap for decision boundary if available -->
                            <Heatmap v-if="heatmapData" :data="heatmapData.z"
                                :x-labels="heatmapData.x.map(v => v.toFixed(1))"
                                :y-labels="heatmapData.y.map(v => v.toFixed(1))"
                                title="Decision Boundary (SVM Decision Function)" colorscale="RdBu"
                                :show-annotations="false" :reverse-y="false" height="500px" class="mt-4" />
                        </div>

                        <Heatmap v-if="confusionMatrixData" :data="confusionMatrixData"
                            :x-labels="['Predicted 0', 'Predicted 1']" :y-labels="['Actual 0', 'Actual 1']"
                            title="Confusion Matrix" colorscale="Greens" :show-annotations="true" height="350px" />
                    </div>
                </template>
            </InteractiveSection>
        </template>

        <!-- Quiz Section -->
        <!-- Requirements: 1.1, 3.1, 3.2, 3.5 -->
        <template #quiz>
            <!-- Loading state -->
            <QuizLoadingState v-if="quizLoading" />

            <!-- Error state with retry -->
            <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />

            <!-- Coming soon state -->
            <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />

            <!-- Quiz component -->
            <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70"
                @quiz-completed="handleQuizComplete" />
        </template>
    </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { SVM } from '~/algorithms/classification/SVM';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { ModuleNavItem } from '~/types/module';
import type { Quiz as QuizType, QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

import { useAsyncData } from '#app';


const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/classification/svm/content').first())

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);
const suggestedModule = ref<ModuleNavItem | null>(null);

// Parameters
const parameters = ref({
    C: 1.0,
    kernel: 'rbf',
    gamma: 'scale',
    dataset: 'binary-classification',
});

// Controls configuration
const controls: ControlConfig[] = [
    {
        id: 'dataset',
        type: 'select',
        label: 'Dataset',
        options: [
            { label: 'Binary Classification', value: 'binary-classification' },
            { label: 'Linearly Separable', value: 'linear-separable' },
            { label: 'Non-linear (Circles)', value: 'circles' },
            { label: 'Non-linear (Moons)', value: 'moons' }
        ],
        tooltip: 'Choose a classification dataset to train on',
        group: 'Data',
    },
    {
        id: 'C',
        type: 'slider',
        label: 'C (Regularization)',
        min: 0.01,
        max: 10,
        step: 0.1,
        tooltip: 'Regularization parameter. Small C = wider margin (more regularization), large C = narrower margin (less regularization).',
        group: 'Model Parameters',
    },
    {
        id: 'kernel',
        type: 'select',
        label: 'Kernel Function',
        options: [
            { label: 'Linear', value: 'linear' },
            { label: 'RBF (Radial Basis Function)', value: 'rbf' },
            { label: 'Polynomial', value: 'poly' }
        ],
        tooltip: 'Kernel function for non-linear classification',
        group: 'Model Parameters',
    },
    {
        id: 'gamma',
        type: 'select',
        label: 'Gamma (RBF/Poly)',
        options: [
            { label: 'Scale (1 / (n_features × X.var()))', value: 'scale' },
            { label: 'Auto (1 / n_features)', value: 'auto' },
            { label: '0.1', value: '0.1' },
            { label: '1.0', value: '1.0' },
            { label: '10.0', value: '10.0' }
        ],
        tooltip: 'Kernel coefficient. Higher gamma = more complex boundary.',
        group: 'Model Parameters',
    },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const heatmapData = ref<{ x: number[], y: number[], z: number[][] } | null>(null);
const confusionMatrixData = ref<number[][] | null>(null);

// Algorithm instance
let algorithm: SVM | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
    moduleStore.setCurrentModule('svm');
    moduleStore.setParameters(parameters.value);

    // Load default dataset
    await loadInitialDataset();

    // Load navigation
    previousModule.value = await getPreviousModule('svm');
    nextModule.value = await getNextModule('svm');

    // For now, suggested module is the same as next module
    suggestedModule.value = nextModule.value;

    // Load quiz data
    // Requirements: 1.1, 2.1, 2.4, 6.1, 6.2
    await loadQuiz('svm', '/content/modules/classification/svm/quiz.json');
});

/**
 * Load initial dataset
 */
const loadInitialDataset = async () => {
    const dataset = await loadDataset(parameters.value.dataset);
    if (dataset) {
        moduleStore.loadDataset(dataset.data);
    }
};

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
    parameters.value[id] = value;
    moduleStore.setParameter(id, value);

    // Reload dataset if changed
    if (id === 'dataset') {
        const dataset = await loadDataset(value);
        if (dataset) {
            moduleStore.loadDataset(dataset.data);
            // Clear previous results
            metrics.value = null;
            scatterSeries.value = [];
            heatmapData.value = null;
            confusionMatrixData.value = null;
        }
    }
};

/**
 * Handle reset
 */
const handleReset = async () => {
    parameters.value = {
        C: 1.0,
        kernel: 'rbf',
        gamma: 'scale',
        dataset: 'binary-classification',
    };
    moduleStore.setParameters(parameters.value);

    await loadInitialDataset();

    metrics.value = null;
    scatterSeries.value = [];
    heatmapData.value = null;
    confusionMatrixData.value = null;
    error.value = null;
};

/**
 * Handle train
 */
const handleTrain = async () => {
    isTraining.value = true;
    trainingProgress.value = 0;
    error.value = null;

    try {
        // Get dataset from store
        const dataset = moduleStore.dataset;
        if (!dataset || dataset.length === 0) {
            throw new Error('No dataset loaded');
        }

        // Prepare data
        const X = dataset.map(d => d.features);
        const y = dataset.map(d => d.label as number);

        // Validate binary labels
        const uniqueLabels = [...new Set(y)];
        if (uniqueLabels.length !== 2) {
            throw new Error('Dataset must have exactly 2 classes');
        }

        // Parse gamma parameter
        let gammaValue: 'scale' | 'auto' | number = parameters.value.gamma as any;
        if (parameters.value.gamma !== 'scale' && parameters.value.gamma !== 'auto') {
            gammaValue = parseFloat(parameters.value.gamma as string);
        }

        // Create and train algorithm
        algorithm = new SVM({
            C: parameters.value.C,
            kernel: parameters.value.kernel as 'linear' | 'rbf' | 'poly',
            gamma: gammaValue,
            maxIter: 1000,
            tol: 1e-3
        });

        // Train with progress updates
        const startTime = Date.now();
        await algorithm.fit(X, y);
        const endTime = Date.now();

        console.log(`Training completed in ${endTime - startTime}ms`);

        // Get results
        metrics.value = algorithm.getMetrics();
        const visualizations = algorithm.getVisualizationData();

        // Set confusion matrix data
        if (metrics.value.confusionMatrix) {
            confusionMatrixData.value = metrics.value.confusionMatrix;
        }

        // Process visualizations
        if (visualizations.length > 0) {
            // Scatter plot with classes and support vectors
            const scatterViz = visualizations[0];
            if (scatterViz.type === 'scatter2d' && scatterViz.data) {
                const series: ScatterSeries[] = [];

                // Add class 0 points
                if (scatterViz.data.class0) {
                    series.push({
                        x: scatterViz.data.class0.x,
                        y: scatterViz.data.class0.y,
                        name: scatterViz.data.class0.name || 'Class 0',
                        color: scatterViz.data.class0.marker?.color || '#3b82f6',
                        mode: 'markers',
                        marker: {
                            size: scatterViz.data.class0.marker?.size || 8,
                            symbol: scatterViz.data.class0.marker?.symbol || 'circle',
                        }
                    });
                }

                // Add class 1 points
                if (scatterViz.data.class1) {
                    series.push({
                        x: scatterViz.data.class1.x,
                        y: scatterViz.data.class1.y,
                        name: scatterViz.data.class1.name || 'Class 1',
                        color: scatterViz.data.class1.marker?.color || '#ef4444',
                        mode: 'markers',
                        marker: {
                            size: scatterViz.data.class1.marker?.size || 8,
                            symbol: scatterViz.data.class1.marker?.symbol || 'x',
                        }
                    });
                }

                // Add support vectors
                if (scatterViz.data.supportVectors) {
                    series.push({
                        x: scatterViz.data.supportVectors.x,
                        y: scatterViz.data.supportVectors.y,
                        name: scatterViz.data.supportVectors.name || 'Support Vectors',
                        color: scatterViz.data.supportVectors.marker?.color || 'rgba(0, 0, 0, 0)',
                        mode: 'markers',
                        marker: {
                            size: scatterViz.data.supportVectors.marker?.size || 14,
                            symbol: scatterViz.data.supportVectors.marker?.symbol || 'circle-open',
                            line: scatterViz.data.supportVectors.marker?.line || { color: '#000000', width: 2 }
                        }
                    });
                }

                scatterSeries.value = series;
            }

            // Heatmap for decision boundary
            const heatmapViz = visualizations.find(v => v.type === 'heatmap');
            if (heatmapViz && heatmapViz.data) {
                heatmapData.value = {
                    x: heatmapViz.data.x,
                    y: heatmapViz.data.y,
                    z: heatmapViz.data.z
                };
            }
        }

        trainingProgress.value = 1;
    } catch (err) {
        console.error('Training error:', err);
        error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
    } finally {
        isTraining.value = false;
    }
};

/**
 * Handle retry
 */
const handleRetry = () => {
    error.value = null;
    handleTrain();
};

/**
 * Handle quiz completion
 * Processes quiz results and verifies progress store integration
 * 
 * Requirements: 3.3, 3.4, 5.1, 5.2, 5.4
 */
const handleQuizComplete = (result: QuizResult) => {
    console.log('Quiz completed:', result);

    // The Quiz component already handles:
    // - Saving score to progress store (Requirement 5.1, 5.2)
    // - Marking module as complete if score >= 70% (Requirement 3.4, 5.4)
    // - Tracking analytics (Requirement 3.3)

    // Optional: Show success message or navigate to next module
    if (result.score >= 70) {
        console.log('Quiz passed! Module marked as complete.');
    }
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);
</script>
