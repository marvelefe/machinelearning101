/**
 * Named Entity Recognition (NER) Algorithm
 * 
 * Implements entity extraction and classification using a rule-based approach
 * combined with pattern matching. Identifies and classifies named entities such as
 * persons, organizations, locations, dates, and other entity types.
 * 
 * Requirements: 3.2, 3.4, 3.5
 */

import { BaseAlgorithm, type VisualizationData, type Metrics } from '~/types/algorithm';

export interface NEROptions {
  entityTypes: string[]; // Types of entities to recognize
  caseSensitive: boolean;
  useContextWindow: boolean;
  contextWindowSize: number;
}

export interface Entity {
  text: string;
  type: string;
  start: number;
  end: number;
  confidence: number;
}

export interface NERPrediction {
  text: string;
  entities: Entity[];
  highlightedText: string;
}

/**
 * Named Entity Recognition using pattern matching and rules
 * Requirements: 3.2, 3.4
 */
export class NamedEntityRecognition extends BaseAlgorithm {
  name = 'Named Entity Recognition';
  
  private options: NEROptions;
  private entityPatterns: Map<string, RegExp[]> = new Map();
  private entityGazetteer: Map<string, Set<string>> = new Map(); // Known entities by type
  private trainingTexts: string[] = [];
  private trainingEntities: Entity[][] = [];
  private predictions: NERPrediction[] = [];
  
  // Pre-defined patterns for common entity types
  private readonly defaultPatterns = {
    PERSON: [
      /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b/g, // John Smith
      /\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g // Dr. Jane Doe
    ],
    ORGANIZATION: [
      /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Group))\.?)\b/g,
      /\b([A-Z][A-Z]+)\b/g // Acronyms like NASA, FBI
    ],
    LOCATION: [
      /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|Province|County|District)))\b/g,
      /\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g // "in New York"
    ],
    DATE: [
      /\b(\d{1,2}\/\d{1,2}\/\d{2,4})\b/g, // 12/31/2023
      /\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b/g,
      /\b(\d{4}-\d{2}-\d{2})\b/g // 2023-12-31
    ],
    TIME: [
      /\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?)\b/g
    ],
    MONEY: [
      /\b(\$\d+(?:,\d{3})*(?:\.\d{2})?)\b/g,
      /\b(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP))\b/g
    ],
    EMAIL: [
      /\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b/g
    ],
    PHONE: [
      /\b(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b/g
    ]
  };
  
  constructor(options: Partial<NEROptions> = {}) {
    super();
    this.options = {
      entityTypes: options.entityTypes ?? ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE'],
      caseSensitive: options.caseSensitive ?? true,
      useContextWindow: options.useContextWindow ?? true,
      contextWindowSize: options.contextWindowSize ?? 5
    };
    
    // Initialize patterns for selected entity types
    this.initializePatterns();
  }
  
  /**
   * Initialize entity patterns based on selected types
   * Requirements: 3.4
   */
  private initializePatterns(): void {
    for (const entityType of this.options.entityTypes) {
      const patterns = this.defaultPatterns[entityType as keyof typeof this.defaultPatterns];
      if (patterns) {
        this.entityPatterns.set(entityType, patterns);
      }
      this.entityGazetteer.set(entityType, new Set());
    }
  }
  
  /**
   * Train NER model on annotated text data
   * Requirements: 3.2, 3.4
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    throw new Error('Use trainOnAnnotatedText() method for NER instead of fit()');
  }
  
  /**
   * Train on annotated text corpus
   * Requirements: 3.2, 3.4
   * 
   * @param texts - Array of text samples
   * @param entities - Array of entity annotations for each text
   */
  async trainOnAnnotatedText(texts: string[], entities: Entity[][]): Promise<void> {
    if (texts.length !== entities.length) {
      throw new Error('Number of texts must match number of entity annotations');
    }
    
    if (texts.length === 0) {
      throw new Error('Training data cannot be empty');
    }
    
    this.trainingTexts = texts;
    this.trainingEntities = entities;
    
    // Build gazetteer from training data
    this.buildGazetteer(entities);
    
    // Learn patterns from training data
    this.learnPatterns(texts, entities);
    
    this.trained = true;
  }
  
  /**
   * Build gazetteer (dictionary) of known entities
   * Requirements: 3.4
   */
  private buildGazetteer(entitiesArray: Entity[][]): void {
    for (const entities of entitiesArray) {
      for (const entity of entities) {
        const gazetteer = this.entityGazetteer.get(entity.type);
        if (gazetteer) {
          const key = this.options.caseSensitive ? entity.text : entity.text.toLowerCase();
          gazetteer.add(key);
        }
      }
    }
  }
  
  /**
   * Learn additional patterns from training data
   * Requirements: 3.4
   */
  private learnPatterns(texts: string[], entitiesArray: Entity[][]): void {
    // Extract context patterns around entities
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const entities = entitiesArray[i];
      
      for (const entity of entities) {
        // Extract words before and after entity
        const before = text.substring(Math.max(0, entity.start - 50), entity.start).trim();
        const after = text.substring(entity.end, Math.min(text.length, entity.end + 50)).trim();
        
        // Store context patterns (simplified - in production would use more sophisticated methods)
        // This is a placeholder for pattern learning
      }
    }
  }
  
  /**
   * Predict entities in new texts
   * Requirements: 3.2, 3.4
   */
  predict(X: number[][]): string[] {
    throw new Error('Use extractEntities() method for NER');
  }
  
  /**
   * Extract entities from text samples
   * Requirements: 3.2, 3.4, 3.5
   */
  extractEntities(texts: string[]): NERPrediction[] {
    this.predictions = texts.map(text => {
      const entities = this.extractEntitiesFromText(text);
      const highlightedText = this.createHighlightedText(text, entities);
      
      return {
        text,
        entities,
        highlightedText
      };
    });
    
    return this.predictions;
  }
  
  /**
   * Extract entities from a single text
   * Requirements: 3.2, 3.4
   */
  private extractEntitiesFromText(text: string): Entity[] {
    const entities: Entity[] = [];
    const foundSpans = new Set<string>(); // Track found spans to avoid duplicates
    
    // Apply pattern-based extraction for each entity type
    for (const [entityType, patterns] of this.entityPatterns.entries()) {
      for (const pattern of patterns) {
        // Reset regex
        pattern.lastIndex = 0;
        
        let match;
        while ((match = pattern.exec(text)) !== null) {
          const matchedText = match[1] || match[0];
          const start = match.index + (match[0].indexOf(matchedText));
          const end = start + matchedText.length;
          const spanKey = `${start}-${end}`;
          
          // Skip if already found
          if (foundSpans.has(spanKey)) {
            continue;
          }
          
          // Calculate confidence based on pattern match and gazetteer
          let confidence = 0.6; // Base confidence for pattern match
          
          // Check gazetteer
          const gazetteer = this.entityGazetteer.get(entityType);
          if (gazetteer) {
            const key = this.options.caseSensitive ? matchedText : matchedText.toLowerCase();
            if (gazetteer.has(key)) {
              confidence = 0.9; // Higher confidence if in gazetteer
            }
          }
          
          // Adjust confidence based on context
          if (this.options.useContextWindow) {
            confidence = this.adjustConfidenceByContext(text, start, end, entityType, confidence);
          }
          
          entities.push({
            text: matchedText,
            type: entityType,
            start,
            end,
            confidence
          });
          
          foundSpans.add(spanKey);
        }
      }
    }
    
    // Apply gazetteer-based extraction
    for (const [entityType, gazetteer] of this.entityGazetteer.entries()) {
      for (const knownEntity of gazetteer) {
        const searchText = this.options.caseSensitive ? text : text.toLowerCase();
        const searchEntity = this.options.caseSensitive ? knownEntity : knownEntity.toLowerCase();
        
        let index = 0;
        while ((index = searchText.indexOf(searchEntity, index)) !== -1) {
          const start = index;
          const end = index + knownEntity.length;
          const spanKey = `${start}-${end}`;
          
          if (!foundSpans.has(spanKey)) {
            entities.push({
              text: text.substring(start, end),
              type: entityType,
              start,
              end,
              confidence: 0.85
            });
            foundSpans.add(spanKey);
          }
          
          index = end;
        }
      }
    }
    
    // Sort entities by start position
    entities.sort((a, b) => a.start - b.start);
    
    // Resolve overlapping entities (keep higher confidence)
    return this.resolveOverlaps(entities);
  }
  
  /**
   * Adjust confidence based on surrounding context
   * Requirements: 3.4
   */
  private adjustConfidenceByContext(
    text: string,
    start: number,
    end: number,
    entityType: string,
    baseConfidence: number
  ): number {
    const windowSize = this.options.contextWindowSize;
    
    // Extract context words
    const beforeText = text.substring(Math.max(0, start - 100), start);
    const afterText = text.substring(end, Math.min(text.length, end + 100));
    
    const beforeWords = beforeText.toLowerCase().split(/\s+/).slice(-windowSize);
    const afterWords = afterText.toLowerCase().split(/\s+/).slice(0, windowSize);
    
    // Context indicators for different entity types
    const contextIndicators: { [key: string]: { before: string[]; after: string[] } } = {
      PERSON: {
        before: ['mr', 'mrs', 'ms', 'dr', 'prof', 'president', 'ceo'],
        after: ['said', 'stated', 'announced', 'reported']
      },
      ORGANIZATION: {
        before: ['at', 'for', 'with'],
        after: ['inc', 'corp', 'llc', 'announced', 'reported']
      },
      LOCATION: {
        before: ['in', 'at', 'from', 'to', 'near'],
        after: ['city', 'state', 'country']
      },
      DATE: {
        before: ['on', 'since', 'until', 'by'],
        after: []
      }
    };
    
    const indicators = contextIndicators[entityType];
    if (!indicators) {
      return baseConfidence;
    }
    
    let boost = 0;
    
    // Check before context
    for (const word of beforeWords) {
      if (indicators.before.includes(word)) {
        boost += 0.1;
      }
    }
    
    // Check after context
    for (const word of afterWords) {
      if (indicators.after.includes(word)) {
        boost += 0.1;
      }
    }
    
    return Math.min(0.99, baseConfidence + boost);
  }
  
  /**
   * Resolve overlapping entities by keeping higher confidence ones
   * Requirements: 3.4
   */
  private resolveOverlaps(entities: Entity[]): Entity[] {
    if (entities.length <= 1) {
      return entities;
    }
    
    const resolved: Entity[] = [];
    let current = entities[0];
    
    for (let i = 1; i < entities.length; i++) {
      const next = entities[i];
      
      // Check for overlap
      if (next.start < current.end) {
        // Keep entity with higher confidence
        if (next.confidence > current.confidence) {
          current = next;
        }
        // Otherwise keep current and skip next
      } else {
        // No overlap, add current and move to next
        resolved.push(current);
        current = next;
      }
    }
    
    // Add last entity
    resolved.push(current);
    
    return resolved;
  }
  
  /**
   * Create highlighted text with entity markers
   * Requirements: 3.5
   */
  private createHighlightedText(text: string, entities: Entity[]): string {
    if (entities.length === 0) {
      return text;
    }
    
    let highlighted = '';
    let lastIndex = 0;
    
    for (const entity of entities) {
      // Add text before entity
      highlighted += text.substring(lastIndex, entity.start);
      
      // Add highlighted entity
      highlighted += `<mark class="entity-${entity.type.toLowerCase()}" data-confidence="${entity.confidence.toFixed(2)}">${entity.text}</mark>`;
      
      lastIndex = entity.end;
    }
    
    // Add remaining text
    highlighted += text.substring(lastIndex);
    
    return highlighted;
  }
  
  /**
   * Add custom entity pattern
   * Requirements: 3.4
   */
  addPattern(entityType: string, pattern: RegExp): void {
    if (!this.entityPatterns.has(entityType)) {
      this.entityPatterns.set(entityType, []);
      this.entityGazetteer.set(entityType, new Set());
    }
    
    const patterns = this.entityPatterns.get(entityType)!;
    patterns.push(pattern);
  }
  
  /**
   * Add entity to gazetteer
   * Requirements: 3.4
   */
  addToGazetteer(entityType: string, entity: string): void {
    if (!this.entityGazetteer.has(entityType)) {
      this.entityGazetteer.set(entityType, new Set());
    }
    
    const gazetteer = this.entityGazetteer.get(entityType)!;
    const key = this.options.caseSensitive ? entity : entity.toLowerCase();
    gazetteer.add(key);
  }
  
  /**
   * Get visualization data for entity highlighting
   * Requirements: 3.5
   */
  getVisualizationData(): VisualizationData | VisualizationData[] {
    const visualizations: VisualizationData[] = [];
    
    // Entity type distribution
    if (this.predictions.length > 0) {
      const entityTypeCounts: { [key: string]: number } = {};
      for (const pred of this.predictions) {
        for (const entity of pred.entities) {
          entityTypeCounts[entity.type] = (entityTypeCounts[entity.type] || 0) + 1;
        }
      }
      
      if (Object.keys(entityTypeCounts).length > 0) {
        const entityTypes = Object.keys(entityTypeCounts);
        const counts = Object.values(entityTypeCounts);
        
        visualizations.push({
          type: 'scatter2d',
          title: 'Entity Type Distribution',
          data: {},
          config: {
            title: 'Entity Type Distribution',
            xLabel: 'Entity Type',
            yLabel: 'Count',
            showLegend: false,
            series: [{
              x: entityTypes.map((_, i) => i),
              y: counts,
              name: 'Entity Count',
              mode: 'markers',
              marker: {
                size: 12,
                symbol: 'square',
              }
            }]
          }
        });
      }
      
      // Confidence distribution
      const confidenceBuckets = [0, 0, 0, 0, 0]; // 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
      const confidenceLabels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'];
      
      for (const pred of this.predictions) {
        for (const entity of pred.entities) {
          const bucket = Math.min(4, Math.floor(entity.confidence * 5));
          confidenceBuckets[bucket]++;
        }
      }
      
      visualizations.push({
        type: 'line',
        title: 'Entity Confidence Distribution',
        data: {},
        config: {
          title: 'Entity Confidence Distribution',
          xLabel: 'Confidence Range',
          yLabel: 'Count',
          showLegend: false,
          series: [{
            x: confidenceLabels.map((_, i) => i),
            y: confidenceBuckets,
            name: 'Entity Count',
            color: '#6366f1',
            showMarkers: true,
          }]
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Get NER metrics
   * Requirements: 3.4
   */
  getMetrics(): Metrics {
    const metrics: Metrics = {
      entityTypes: this.options.entityTypes.length,
      gazetteerSize: Array.from(this.entityGazetteer.values())
        .reduce((sum, set) => sum + set.size, 0),
      patternsCount: Array.from(this.entityPatterns.values())
        .reduce((sum, patterns) => sum + patterns.length, 0)
    };
    
    // Calculate metrics from predictions
    if (this.predictions.length > 0) {
      let totalEntities = 0;
      let totalConfidence = 0;
      const entityTypeCount: { [key: string]: number } = {};
      
      for (const pred of this.predictions) {
        totalEntities += pred.entities.length;
        for (const entity of pred.entities) {
          totalConfidence += entity.confidence;
          entityTypeCount[entity.type] = (entityTypeCount[entity.type] || 0) + 1;
        }
      }
      
      metrics.totalEntitiesExtracted = totalEntities;
      metrics.averageConfidence = totalEntities > 0 ? totalConfidence / totalEntities : 0;
      metrics.averageEntitiesPerText = totalEntities / this.predictions.length;
      metrics.entityTypeDistribution = entityTypeCount;
    }
    
    // If we have training data, calculate precision/recall
    if (this.trainingTexts.length > 0 && this.predictions.length > 0) {
      const { precision, recall, f1 } = this.calculatePRF();
      metrics.precision = precision;
      metrics.recall = recall;
      metrics.f1Score = f1;
    }
    
    return metrics;
  }
  
  /**
   * Calculate precision, recall, and F1 score
   * Requirements: 3.4
   */
  private calculatePRF(): { precision: number; recall: number; f1: number } {
    // Compare predictions against training data
    let truePositives = 0;
    let falsePositives = 0;
    let falseNegatives = 0;
    
    const numSamples = Math.min(this.predictions.length, this.trainingEntities.length);
    
    for (let i = 0; i < numSamples; i++) {
      const predicted = this.predictions[i].entities;
      const actual = this.trainingEntities[i];
      
      // Create sets of entity spans for comparison
      const predictedSpans = new Set(
        predicted.map(e => `${e.start}-${e.end}-${e.type}`)
      );
      const actualSpans = new Set(
        actual.map(e => `${e.start}-${e.end}-${e.type}`)
      );
      
      // Count matches
      for (const span of predictedSpans) {
        if (actualSpans.has(span)) {
          truePositives++;
        } else {
          falsePositives++;
        }
      }
      
      for (const span of actualSpans) {
        if (!predictedSpans.has(span)) {
          falseNegatives++;
        }
      }
    }
    
    const precision = truePositives + falsePositives > 0
      ? truePositives / (truePositives + falsePositives)
      : 0;
    
    const recall = truePositives + falseNegatives > 0
      ? truePositives / (truePositives + falseNegatives)
      : 0;
    
    const f1 = precision + recall > 0
      ? 2 * (precision * recall) / (precision + recall)
      : 0;
    
    return { precision, recall, f1 };
  }
  
  /**
   * Reset NER state
   */
  override reset(): void {
    super.reset();
    this.trainingTexts = [];
    this.trainingEntities = [];
    this.predictions = [];
    
    // Reset gazetteers but keep patterns
    for (const gazetteer of this.entityGazetteer.values()) {
      gazetteer.clear();
    }
  }
}
