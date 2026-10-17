/**
 * Composable for managing module metadata and navigation
 */

import type { ModuleConfig, ModuleMetadata, DomainInfo, Domain, ModuleNavItem } from '~/types/module';
import { useProgressStore } from '~/stores/progress';

/**
 * Domain definitions with metadata
 * Requirements: 1.1, 1.2, 2.1, 3.1, 4.1, 12.1
 */
export const DOMAINS: Record<Domain, Omit<DomainInfo, 'modules'>> = {
  regression: {
    id: 'regression',
    name: 'Regression',
    description: 'Learn how to predict continuous values using regression algorithms',
    icon: 'healthicons:chart-line',
    color: 'blue',
  },
  classification: {
    id: 'classification',
    name: 'Classification',
    description: 'Discover how to categorize data into distinct classes',
    icon: 'healthicons:ui-menu-grid',
    color: 'green',
  },
  clustering: {
    id: 'clustering',
    name: 'Clustering',
    description: 'Explore unsupervised learning to find patterns in data',
    icon: 'gravity-ui:circles-5-random',
    color: 'purple',
  },
  preprocessing: {
    id: 'preprocessing',
    name: 'Preprocessing',
    description: 'Master data preparation techniques for better model performance',
    icon: 'material-symbols:filter-alt',
    color: 'orange',
  },
  optimization: {
    id: 'optimization',
    name: 'Optimization',
    description: 'Understand how algorithms learn through optimization techniques',
    icon: 'ion:md-trending-up',
    color: 'red',
  },
  'computer-vision': {
    id: 'computer-vision',
    name: 'Computer Vision',
    description: 'Learn how machines interpret and understand visual information',
    icon: 'material-symbols:image-search',
    color: 'cyan',
  },
  nlp: {
    id: 'nlp',
    name: 'Natural Language Processing',
    description: 'Discover how machines understand and generate human language',
    icon: 'material-symbols:chat-bubble',
    color: 'indigo',
  },
  'reinforcement-learning': {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    description: 'Explore how agents learn through interaction and rewards',
    icon: 'game-icons:brain',
    color: 'pink',
  },
    advanced: {
    id: 'advanced',
    name: 'Advanced Topics',
    description: 'Dive into advanced machine learning concepts and techniques',
    icon: 'material-symbols:rocket-launch-rounded',
    color: 'gray',
  },
};

/**
 * Module registry - maps module IDs to their configurations
 * This is a static registry that should match the content structure
 * Requirements: 12.2, 12.3, 12.4
 */
const MODULE_REGISTRY: Record<string, { domain: Domain; path: string }> = {
  // Regression Domain (5 modules)
  'linear-regression': { domain: 'regression', path: '/modules/regression/linear-regression' },
  'polynomial-regression': { domain: 'regression', path: '/modules/regression/polynomial-regression' },
  'ridge-regression': { domain: 'regression', path: '/modules/regression/ridge-regression' },
  'lasso-regression': { domain: 'regression', path: '/modules/regression/lasso-regression' },
  'elastic-net-regression': { domain: 'regression', path: '/modules/regression/elastic-net-regression' },

  // Classification Domain (9 modules)
  'logistic-regression': { domain: 'classification', path: '/modules/classification/logistic-regression' },
  'knn': { domain: 'classification', path: '/modules/classification/knn' },
  'decision-trees': { domain: 'classification', path: '/modules/classification/decision-trees' },
  'svm': { domain: 'classification', path: '/modules/classification/svm' },
  'random-forest': { domain: 'classification', path: '/modules/classification/random-forest' },
  'naive-bayes': { domain: 'classification', path: '/modules/classification/naive-bayes' },
  'gradient-boosting': { domain: 'classification', path: '/modules/classification/gradient-boosting' },
  'adaboost': { domain: 'classification', path: '/modules/classification/adaboost' },
  'multi-class-classification': { domain: 'classification', path: '/modules/classification/multi-class-classification' },

  // Clustering Domain (7 modules)
  'k-means': { domain: 'clustering', path: '/modules/clustering/k-means' },
  'hierarchical-clustering': { domain: 'clustering', path: '/modules/clustering/hierarchical-clustering' },
  'dbscan': { domain: 'clustering', path: '/modules/clustering/dbscan' },
  'gaussian-mixture-models': { domain: 'clustering', path: '/modules/clustering/gaussian-mixture-models' },
  'mean-shift': { domain: 'clustering', path: '/modules/clustering/mean-shift' },
  'spectral-clustering': { domain: 'clustering', path: '/modules/clustering/spectral-clustering' },
  'cluster-evaluation': { domain: 'clustering', path: '/modules/clustering/cluster-evaluation' },

  // Preprocessing Domain (7 modules)
  'feature-scaling': { domain: 'preprocessing', path: '/modules/preprocessing/feature-scaling' },
  'missing-value-imputation': { domain: 'preprocessing', path: '/modules/preprocessing/missing-value-imputation' },
  'feature-encoding': { domain: 'preprocessing', path: '/modules/preprocessing/feature-encoding' },
  'feature-selection': { domain: 'preprocessing', path: '/modules/preprocessing/feature-selection' },
  'dimensionality-reduction': { domain: 'preprocessing', path: '/modules/preprocessing/dimensionality-reduction' },
  'outlier-detection': { domain: 'preprocessing', path: '/modules/preprocessing/outlier-detection' },
  'data-augmentation': { domain: 'preprocessing', path: '/modules/preprocessing/data-augmentation' },

  // Optimization Domain (7 modules)
  'gradient-descent': { domain: 'optimization', path: '/modules/optimization/gradient-descent' },
  'sgd': { domain: 'optimization', path: '/modules/optimization/sgd' },
  'adam-optimizer': { domain: 'optimization', path: '/modules/optimization/adam-optimizer' },
  'rmsprop-optimizer': { domain: 'optimization', path: '/modules/optimization/rmsprop-optimizer' },
  'learning-rate-scheduling': { domain: 'optimization', path: '/modules/optimization/learning-rate-scheduling' },
  'momentum-nesterov': { domain: 'optimization', path: '/modules/optimization/momentum-nesterov' },

  // Computer Vision Domain (5 modules)
  'image-classification': { domain: 'computer-vision', path: '/modules/computer-vision/image-classification' },
  'cnn': { domain: 'computer-vision', path: '/modules/computer-vision/cnn' },
  'object-detection': { domain: 'computer-vision', path: '/modules/computer-vision/object-detection' },
  'image-segmentation': { domain: 'computer-vision', path: '/modules/computer-vision/image-segmentation' },
  'transfer-learning': { domain: 'computer-vision', path: '/modules/computer-vision/transfer-learning' },

  // Natural Language Processing Domain (5 modules)
  'text-preprocessing': { domain: 'nlp', path: '/modules/nlp/text-preprocessing' },
  'word-embeddings': { domain: 'nlp', path: '/modules/nlp/word-embeddings' },
  'sentiment-analysis': { domain: 'nlp', path: '/modules/nlp/sentiment-analysis' },
  'named-entity-recognition': { domain: 'nlp', path: '/modules/nlp/named-entity-recognition' },
  'text-generation': { domain: 'nlp', path: '/modules/nlp/text-generation' },

  // Reinforcement Learning Domain (5 modules)
  'q-learning': { domain: 'reinforcement-learning', path: '/modules/reinforcement-learning/q-learning' },
  'deep-q-networks': { domain: 'reinforcement-learning', path: '/modules/reinforcement-learning/deep-q-networks' },
  'policy-gradients': { domain: 'reinforcement-learning', path: '/modules/reinforcement-learning/policy-gradients' },
  'actor-critic': { domain: 'reinforcement-learning', path: '/modules/reinforcement-learning/actor-critic' },
  'multi-armed-bandits': { domain: 'reinforcement-learning', path: '/modules/reinforcement-learning/multi-armed-bandits' },

  // Advanced Domain (8 modules)
  'pca': { domain: 'advanced', path: '/modules/advanced/pca' },
  'tsne': { domain: 'advanced', path: '/modules/advanced/tsne' },
  'neural-network': { domain: 'advanced', path: '/modules/advanced/neural-network' },
  'autoencoders': { domain: 'advanced', path: '/modules/advanced/autoencoders' },
  'gan': { domain: 'advanced', path: '/modules/advanced/gan' },
  'attention-mechanisms': { domain: 'advanced', path: '/modules/advanced/attention-mechanisms' },
  'hyperparameter-tuning': { domain: 'advanced', path: '/modules/advanced/hyperparameter-tuning' },
  'model-ensembling': { domain: 'advanced', path: '/modules/advanced/model-ensembling' },
};

export function useModules() {
  const progressStore = useProgressStore();

  /**
   * Load module configuration from content directory
   */
  const loadModuleConfig = async (moduleId: string): Promise<ModuleConfig | null> => {
    try {
      const registryEntry = MODULE_REGISTRY[moduleId];
      if (!registryEntry) {
        console.warn(`Module ${moduleId} not found in registry`);
        return null;
      }

      // In a real implementation, this would fetch from the content directory
      // For now, we'll construct the path and use dynamic import
      const configPath = `/content/modules/${registryEntry.domain}/${moduleId}/config.json`;
      
      // Use fetch to load the config
      const response = await fetch(configPath);
      if (!response.ok) {
        console.warn(`Failed to load config for ${moduleId}`);
        return null;
      }
      
      const config: ModuleConfig = await response.json();
      return config;
    } catch (error) {
      console.error(`Error loading module config for ${moduleId}:`, error);
      return null;
    }
  };

  /**
   * Get all modules for a specific domain
   */
  const getModulesByDomain = async (domain: Domain): Promise<ModuleMetadata[]> => {
    const moduleIds = Object.entries(MODULE_REGISTRY)
      .filter(([_, entry]) => entry.domain === domain)
      .map(([id]) => id);

    // Load all configs concurrently instead of sequentially
    const configPromises = moduleIds.map(async (moduleId) => {
      const config = await loadModuleConfig(moduleId);
      if (!config) return null;

        const registryEntry = MODULE_REGISTRY[moduleId];
      return {
          ...config,
          path: registryEntry.path,
          completed: progressStore.isModuleCompleted(moduleId),
          quizScore: progressStore.getQuizScore(moduleId),
      };
    });

    const results = await Promise.all(configPromises);
    
    // Filter out null results
    return results.filter((module): module is NonNullable<typeof module> => module !== null);
  };

  /**
   * Get domain information with all its modules
   */
  const getDomainInfo = async (domain: Domain): Promise<DomainInfo> => {
    const domainMeta = DOMAINS[domain];
    const modules = await getModulesByDomain(domain);

    return {
      ...domainMeta,
      modules,
    };
  };

  /**
   * Get all domains with their modules
   */
  const getAllDomains = async (): Promise<DomainInfo[]> => {
    const domainKeys = Object.keys(DOMAINS) as Domain[];
    
    // Use Promise.all to load all domains concurrently
    const domainPromises = domainKeys.map(domain => getDomainInfo(domain));
    
    return Promise.all(domainPromises);
  };

  /**
   * Get module metadata by ID
   */
  const getModuleMetadata = async (moduleId: string): Promise<ModuleMetadata | null> => {
    const config = await loadModuleConfig(moduleId);
    if (!config) return null;

    const registryEntry = MODULE_REGISTRY[moduleId];
    if (!registryEntry) return null;

    return {
      ...config,
      path: registryEntry.path,
      completed: progressStore.isModuleCompleted(moduleId),
      quizScore: progressStore.getQuizScore(moduleId),
    };
  };

  /**
   * Get navigation items for all modules
   */
  const getAllModuleNavItems = async (): Promise<ModuleNavItem[]> => {
    const items: ModuleNavItem[] = [];

    for (const [moduleId, registryEntry] of Object.entries(MODULE_REGISTRY)) {
      const config = await loadModuleConfig(moduleId);
      if (config) {
        items.push({
          id: moduleId,
          name: config.name,
          path: registryEntry.path,
          domain: registryEntry.domain,
          difficulty: config.difficulty,
          completed: progressStore.isModuleCompleted(moduleId),
        });
      }
    }

    return items;
  };

  /**
   * Get next suggested module based on current module
   * Optimized to only load configs for current, next module
   */
  const getNextModule = async (currentModuleId: string): Promise<ModuleNavItem | null> => {
    // Get all module IDs in order (no config loading)
    const moduleIds = Object.keys(MODULE_REGISTRY);
    const currentIndex = moduleIds.findIndex(id => id === currentModuleId);
    
    if (currentIndex === -1 || currentIndex === moduleIds.length - 1) {
      return null;
    }

    // Only load config for the next module
    const nextModuleId = moduleIds[currentIndex + 1];
    const config = await loadModuleConfig(nextModuleId);
    if (!config) return null;

    const registryEntry = MODULE_REGISTRY[nextModuleId];
    return {
      id: nextModuleId,
      name: config.name,
      path: registryEntry.path,
      domain: registryEntry.domain,
      difficulty: config.difficulty,
      completed: progressStore.isModuleCompleted(nextModuleId),
    };
  };

  /**
   * Get previous module based on current module
   * Optimized to only load configs for current, previous module
   */
  const getPreviousModule = async (currentModuleId: string): Promise<ModuleNavItem | null> => {
    // Get all module IDs in order (no config loading)
    const moduleIds = Object.keys(MODULE_REGISTRY);
    const currentIndex = moduleIds.findIndex(id => id === currentModuleId);
    
    if (currentIndex <= 0) {
      return null;
    }

    // Only load config for the previous module
    const prevModuleId = moduleIds[currentIndex - 1];
    const config = await loadModuleConfig(prevModuleId);
    if (!config) return null;

    const registryEntry = MODULE_REGISTRY[prevModuleId];
    return {
      id: prevModuleId,
      name: config.name,
      path: registryEntry.path,
      domain: registryEntry.domain,
      difficulty: config.difficulty,
      completed: progressStore.isModuleCompleted(prevModuleId),
    };
  };

  /**
   * Get completion stats for a domain
   */
  const getDomainCompletionStats = async (domain: Domain): Promise<{ completed: number; total: number; percentage: number }> => {
    const modules = await getModulesByDomain(domain);
    const completed = modules.filter(m => m.completed).length;
    const total = modules.length;
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;

    return { completed, total, percentage };
  };

  return {
    DOMAINS,
    MODULE_REGISTRY,
    loadModuleConfig,
    getModulesByDomain,
    getDomainInfo,
    getAllDomains,
    getModuleMetadata,
    getAllModuleNavItems,
    getNextModule,
    getPreviousModule,
    getDomainCompletionStats,
  };
}
