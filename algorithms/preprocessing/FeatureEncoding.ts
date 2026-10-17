/**
 * Feature Encoding Algorithm
 * Implements various encoding techniques for categorical and numerical features
 * Requirements: 8.1, 8.2, 8.3, 8.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm'; 

export type EncodingMethod = 'one-hot' | 'label' | 'ordinal' | 'binary' | 'target' | 'none';

export interface EncodingResult {
  encoded: number[][];
  original: (string | number)[][];
  method: EncodingMethod;
  encodingMap: Map<string, number | number[]>;
  featureNames: string[];
  statistics: {
    originalFeatures: number;
    encodedFeatures: number;
    uniqueValues: number[];
    encodingEfficiency: number;
  };
}

export class FeatureEncoding extends BaseAlgorithm {
  name = 'Feature Encoding';
  
  private data: (string | number)[][] = [];
  private encodedData: number[][] = [];
  private method: EncodingMethod = 'none';
  private encodingMap = new Map<string, number | number[]>();
  private featureNames: string[] = [];
  private targetValues: number[] = [];

  constructor() {
    super();
  }

  /**
   * Set the data to be encoded
   */
  setData(data: (string | number)[][], target?: number[]): void {
    this.data = data;
    this.targetValues = target || [];
    this.encodedData = [];
    this.encodingMap.clear();
    this.featureNames = [];
    this.trained = false;
  }

  /**
   * Fit the encoder and transform the data
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Convert numeric data back to mixed type for encoding
    const mixedData = X.map(row => row.map(val => val));
    this.setData(mixedData, y);
    
    // Apply default one-hot encoding
    this.oneHotEncode();
    this.trained = true;
  }

  /**
   * Apply one-hot encoding to categorical features
   */
  oneHotEncode(): EncodingResult {
    if (this.data.length === 0) {
      throw new Error('No data to encode');
    }

    this.method = 'one-hot';
    const numFeatures = this.data[0].length;
    this.encodingMap.clear();
    this.featureNames = [];

    // Find unique values for each feature
    const uniqueValuesByFeature: Set<string>[] = [];
    for (let j = 0; j < numFeatures; j++) {
      const uniqueValues = new Set<string>();
      this.data.forEach(row => {
        uniqueValues.add(String(row[j]));
      });
      uniqueValuesByFeature.push(uniqueValues);
    }

    // Create encoding map and feature names
    let encodedFeatureIndex = 0;
    for (let j = 0; j < numFeatures; j++) {
      const uniqueValues = Array.from(uniqueValuesByFeature[j]).sort();
      
      if (uniqueValues.length <= 1) {
        // Single value feature - create one binary feature
        this.featureNames.push(`feature_${j}_${uniqueValues[0] || 'empty'}`);
        this.encodingMap.set(`${j}_${uniqueValues[0] || 'empty'}`, encodedFeatureIndex);
        encodedFeatureIndex++;
      } else {
        // Multiple values - create binary feature for each
        uniqueValues.forEach(value => {
          this.featureNames.push(`feature_${j}_${value}`);
          this.encodingMap.set(`${j}_${value}`, encodedFeatureIndex);
          encodedFeatureIndex++;
        });
      }
    }

    // Apply encoding
    this.encodedData = this.data.map(row => {
      const encodedRow = new Array(this.featureNames.length).fill(0);
      
      row.forEach((value, j) => {
        const key = `${j}_${String(value)}`;
        const encodedIndex = this.encodingMap.get(key);
        if (encodedIndex !== undefined) {
          encodedRow[encodedIndex as number] = 1;
        }
      });
      
      return encodedRow;
    });

    return this.getResult();
  }

  /**
   * Apply label encoding to categorical features
   */
  labelEncode(): EncodingResult {
    if (this.data.length === 0) {
      throw new Error('No data to encode');
    }

    this.method = 'label';
    const numFeatures = this.data[0].length;
    this.encodingMap.clear();
    this.featureNames = [];

    // Create label encoding for each feature
    for (let j = 0; j < numFeatures; j++) {
      this.featureNames.push(`feature_${j}_label`);
      
      const uniqueValues = Array.from(new Set(this.data.map(row => String(row[j])))).sort();
      uniqueValues.forEach((value, index) => {
        this.encodingMap.set(`${j}_${value}`, index);
      });
    }

    // Apply encoding
    this.encodedData = this.data.map(row => 
      row.map((value, j) => {
        const key = `${j}_${String(value)}`;
        return this.encodingMap.get(key) as number || 0;
      })
    );

    return this.getResult();
  }

  /**
   * Apply ordinal encoding with custom ordering
   */
  ordinalEncode(ordering?: string[][]): EncodingResult {
    if (this.data.length === 0) {
      throw new Error('No data to encode');
    }

    this.method = 'ordinal';
    const numFeatures = this.data[0].length;
    this.encodingMap.clear();
    this.featureNames = [];

    // Create ordinal encoding for each feature
    for (let j = 0; j < numFeatures; j++) {
      this.featureNames.push(`feature_${j}_ordinal`);
      
      let orderedValues: string[];
      if (ordering && ordering[j]) {
        orderedValues = ordering[j];
      } else {
        // Default: sort unique values
        orderedValues = Array.from(new Set(this.data.map(row => String(row[j])))).sort();
      }
      
      orderedValues.forEach((value, index) => {
        this.encodingMap.set(`${j}_${value}`, index);
      });
    }

    // Apply encoding
    this.encodedData = this.data.map(row => 
      row.map((value, j) => {
        const key = `${j}_${String(value)}`;
        return this.encodingMap.get(key) as number || 0;
      })
    );

    return this.getResult();
  }

  /**
   * Apply binary encoding
   */
  binaryEncode(): EncodingResult {
    if (this.data.length === 0) {
      throw new Error('No data to encode');
    }

    this.method = 'binary';
    const numFeatures = this.data[0].length;
    this.encodingMap.clear();
    this.featureNames = [];

    // First apply label encoding, then convert to binary
    const labelEncoded: number[][] = [];
    const maxValues: number[] = [];

    for (let j = 0; j < numFeatures; j++) {
      const uniqueValues = Array.from(new Set(this.data.map(row => String(row[j])))).sort();
      maxValues[j] = uniqueValues.length - 1;
      
      uniqueValues.forEach((value, index) => {
        this.encodingMap.set(`${j}_${value}`, index);
      });
    }

    // Convert to label encoded first
    const labelEncodedData = this.data.map(row => 
      row.map((value, j) => {
        const key = `${j}_${String(value)}`;
        return this.encodingMap.get(key) as number || 0;
      })
    );

    // Convert each label to binary representation
    let totalBinaryFeatures = 0;
    const binaryBitsPerFeature: number[] = [];
    
    for (let j = 0; j < numFeatures; j++) {
      const bitsNeeded = Math.max(1, Math.ceil(Math.log2(maxValues[j] + 1)));
      binaryBitsPerFeature[j] = bitsNeeded;
      totalBinaryFeatures += bitsNeeded;
      
      for (let bit = 0; bit < bitsNeeded; bit++) {
        this.featureNames.push(`feature_${j}_bit_${bit}`);
      }
    }

    // Apply binary encoding
    this.encodedData = labelEncodedData.map(row => {
      const binaryRow: number[] = [];
      
      row.forEach((value, j) => {
        const bitsNeeded = binaryBitsPerFeature[j];
        const binaryString = value.toString(2).padStart(bitsNeeded, '0');
        
        for (let bit = 0; bit < bitsNeeded; bit++) {
          binaryRow.push(parseInt(binaryString[bit]));
        }
      });
      
      return binaryRow;
    });

    return this.getResult();
  }

  /**
   * Apply target encoding (mean encoding)
   */
  targetEncode(): EncodingResult {
    if (this.data.length === 0) {
      throw new Error('No data to encode');
    }
    
    if (this.targetValues.length === 0) {
      throw new Error('Target values required for target encoding');
    }

    this.method = 'target';
    const numFeatures = this.data[0].length;
    this.encodingMap.clear();
    this.featureNames = [];

    // Calculate mean target value for each category
    for (let j = 0; j < numFeatures; j++) {
      this.featureNames.push(`feature_${j}_target`);
      
      const categoryMeans = new Map<string, number>();
      const categoryCounts = new Map<string, number>();
      
      // Calculate sums and counts
      this.data.forEach((row, i) => {
        const category = String(row[j]);
        const target = this.targetValues[i];
        
        categoryMeans.set(category, (categoryMeans.get(category) || 0) + target);
        categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1);
      });
      
      // Convert sums to means
      categoryMeans.forEach((sum, category) => {
        const count = categoryCounts.get(category) || 1;
        this.encodingMap.set(`${j}_${category}`, sum / count);
      });
    }

    // Apply encoding
    this.encodedData = this.data.map(row => 
      row.map((value, j) => {
        const key = `${j}_${String(value)}`;
        return this.encodingMap.get(key) as number || 0;
      })
    );

    return this.getResult();
  }

  /**
   * Make predictions (transform new data using fitted encoder)
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    // For encoding, prediction means transforming new data
    // Return flattened encoded data for compatibility
    const mixedData = X.map(row => row.map(val => val));
    const tempData = this.data;
    this.data = mixedData;
    
    let result: number[] = [];
    switch (this.method) {
      case 'one-hot':
        result = this.oneHotEncode().encoded.flat();
        break;
      case 'label':
        result = this.labelEncode().encoded.flat();
        break;
      case 'ordinal':
        result = this.ordinalEncode().encoded.flat();
        break;
      case 'binary':
        result = this.binaryEncode().encoded.flat();
        break;
      default:
        result = X.flat();
    }
    
    this.data = tempData;
    return result;
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    const result = this.getResult();
    
    return [
      {
        type: 'heatmap',
        data: {
          z: result.encoded.slice(0, Math.min(20, result.encoded.length)), // Show first 20 rows
          x: result.featureNames,
          y: result.encoded.map((_, i) => `Sample ${i + 1}`).slice(0, Math.min(20, result.encoded.length)),
          colorscale: 'Viridis',
          showscale: true
        },
        layout: {
          title: `${this.method.charAt(0).toUpperCase() + this.method.slice(1)} Encoding Result`,
          xaxis: { title: 'Encoded Features' },
          yaxis: { title: 'Samples' }
        }
      }
    ];
  }

  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    const result = this.getResult();
    
    return {
      originalFeatures: result.statistics.originalFeatures,
      encodedFeatures: result.statistics.encodedFeatures,
      compressionRatio: result.statistics.originalFeatures / result.statistics.encodedFeatures,
      encodingEfficiency: result.statistics.encodingEfficiency,
      uniqueValuesPerFeature: result.statistics.uniqueValues.reduce((a, b) => a + b, 0) / result.statistics.uniqueValues.length
    };
  }

  /**
   * Get the current encoding result
   */
  getResult(): EncodingResult {
    const originalFeatures = this.data[0]?.length || 0;
    const encodedFeatures = this.encodedData[0]?.length || 0;
    
    const uniqueValues = [];
    for (let j = 0; j < originalFeatures; j++) {
      const unique = new Set(this.data.map(row => String(row[j])));
      uniqueValues.push(unique.size);
    }
    
    const encodingEfficiency = originalFeatures > 0 ? encodedFeatures / originalFeatures : 1;
    
    return {
      encoded: this.encodedData.length > 0 ? this.encodedData : this.data.map(row => row.map(val => Number(val) || 0)),
      original: this.data,
      method: this.method,
      encodingMap: this.encodingMap,
      featureNames: this.featureNames,
      statistics: {
        originalFeatures,
        encodedFeatures,
        uniqueValues,
        encodingEfficiency
      }
    };
  }

  /**
   * Reset to initial state
   */
  override reset(): void {
    super.reset();
    this.data = [];
    this.encodedData = [];
    this.method = 'none';
    this.encodingMap.clear();
    this.featureNames = [];
    this.targetValues = [];
  }

  /**
   * Generate sample categorical data for demonstration
   */
  static generateCategoricalData(
    rows: number,
    cols: number,
    categoriesPerFeature: number[] = [3, 4, 2]
  ): (string | number)[][] {
    const categories = [
      ['Red', 'Green', 'Blue', 'Yellow', 'Purple'],
      ['Small', 'Medium', 'Large', 'XLarge'],
      ['A', 'B', 'C', 'D', 'E'],
      ['Low', 'High'],
      ['Cat', 'Dog', 'Bird', 'Fish']
    ];
    
    const data: (string | number)[][] = [];
    
    for (let i = 0; i < rows; i++) {
      const row: (string | number)[] = [];
      for (let j = 0; j < cols; j++) {
        const numCategories = categoriesPerFeature[j] || 3;
        const availableCategories = categories[j % categories.length];
        const categoryIndex = Math.floor(Math.random() * Math.min(numCategories, availableCategories.length));
        row.push(availableCategories[categoryIndex]);
      }
      data.push(row);
    }
    
    return data;
  }
}