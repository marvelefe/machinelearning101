/**
 * SEO Composable
 * 
 * Provides utilities for managing SEO meta tags, Open Graph, Twitter Cards,
 * and structured data (JSON-LD) across the application.
 * 
 * Requirements: 12.6 - SEO optimization with proper meta tags
 */

import { useHead, useSeoMeta } from '#app'
import type { UseSeoMetaInput } from '@unhead/vue'

export interface SEOConfig {
  title: string
  description: string
  keywords?: string
  image?: string
  url?: string
  type?: 'website' | 'article' | 'profile'
  author?: string
  publishedTime?: string
  modifiedTime?: string
  section?: string
  tags?: string[]
  noindex?: boolean
  nofollow?: boolean
}

export interface StructuredDataConfig {
  '@context': string
  '@type': string
  [key: string]: any
}

/**
 * Default SEO configuration
 */
const DEFAULT_SEO = {
  siteName: 'Machine Learning 101',
  siteUrl: 'https://machinelearning101.com',
  defaultImage: 'https://machinelearning101.com/og-image.png',
  // twitterHandle: '@mlsimplified',
  locale: 'en_US',
}

/**
 * Main SEO composable
 */
export function useSEO(config: SEOConfig) {
  const fullTitle = config.title.includes('Machine Learning 101') 
    ? config.title 
    : `${config.title} - Machine Learning 101`
  
  const canonicalUrl = config.url 
    ? `${DEFAULT_SEO.siteUrl}${config.url}` 
    : DEFAULT_SEO.siteUrl
  
  const imageUrl = config.image || DEFAULT_SEO.defaultImage

  // Set meta tags using useSeoMeta
  useSeoMeta({
    title: fullTitle,
    description: config.description,
    keywords: config.keywords,
    
    // Open Graph
    ogTitle: fullTitle,
    ogDescription: config.description,
    ogImage: imageUrl,
    ogUrl: canonicalUrl,
    ogType: config.type || 'website',
    ogSiteName: DEFAULT_SEO.siteName,
    ogLocale: DEFAULT_SEO.locale,
    
    // Twitter Card
    // twitterCard: 'summary_large_image',
    // twitterSite: DEFAULT_SEO.twitterHandle,
    // twitterCreator: DEFAULT_SEO.twitterHandle,
    // twitterTitle: fullTitle,
    // twitterDescription: config.description,
    // twitterImage: imageUrl,
    
    // Article specific (if type is article)
    ...(config.type === 'article' && {
      articleAuthor: config.author,
      articlePublishedTime: config.publishedTime,
      articleModifiedTime: config.modifiedTime,
      articleSection: config.section,
      articleTag: config.tags,
    }),
    
    // Robots
    ...(config.noindex && { robots: 'noindex, nofollow' }),
    // `config.author` is a single string while unhead types `articleAuthor` as
    // a list; keep the runtime value unchanged.
  } as unknown as UseSeoMetaInput)

  // Set canonical URL and other head elements
  useHead({
    title: fullTitle,
    link: [
      {
        rel: 'canonical',
        href: canonicalUrl,
      },
    ],
  })
}

/**
 * Add structured data (JSON-LD) to the page
 */
export function useStructuredData(data: StructuredDataConfig | StructuredDataConfig[]) {
  const structuredData = Array.isArray(data) ? data : [data]
  
  useHead({
    script: structuredData.map(item => ({
      type: 'application/ld+json',
      children: JSON.stringify(item),
    })),
  })
}

/**
 * Generate breadcrumb structured data
 */
export function useBreadcrumbStructuredData(items: Array<{ name: string; url: string }>) {
  const breadcrumbList: StructuredDataConfig = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: items.map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: item.name,
      item: `${DEFAULT_SEO.siteUrl}${item.url}`,
    })),
  }
  
  useStructuredData(breadcrumbList)
}

/**
 * Generate course/learning module structured data
 */
export function useCourseStructuredData(config: {
  name: string
  description: string
  provider: string
  url: string
  difficulty?: string
  timeRequired?: string
  keywords?: string[]
}) {
  const courseData: StructuredDataConfig = {
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: config.name,
    description: config.description,
    provider: {
      '@type': 'Organization',
      name: config.provider,
      sameAs: DEFAULT_SEO.siteUrl,
    },
    url: `${DEFAULT_SEO.siteUrl}${config.url}`,
    ...(config.difficulty && { educationalLevel: config.difficulty }),
    ...(config.timeRequired && { timeRequired: config.timeRequired }),
    ...(config.keywords && { keywords: config.keywords.join(', ') }),
  }
  
  useStructuredData(courseData)
}

/**
 * Generate organization structured data
 */
export function useOrganizationStructuredData() {
  const orgData: StructuredDataConfig = {
    '@context': 'https://schema.org',
    '@type': 'Organization',
    name: DEFAULT_SEO.siteName,
    url: DEFAULT_SEO.siteUrl,
    logo: `${DEFAULT_SEO.siteUrl}/logo.png`,
    // sameAs: [
    //   'https://github.com/marvelefe/machinelearning101', 
    // ],
    contactPoint: {
      '@type': 'ContactPoint',
      email: 'hello@machinelearning101.com',
      contactType: 'Customer Service',
    },
  }
  
  useStructuredData(orgData)
}

/**
 * Generate website structured data
 */
export function useWebsiteStructuredData() {
  const websiteData: StructuredDataConfig = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: DEFAULT_SEO.siteName,
    url: DEFAULT_SEO.siteUrl,
    description: 'Learn machine learning through interactive visualizations and hands-on experimentation',
    potentialAction: {
      '@type': 'SearchAction',
      target: `${DEFAULT_SEO.siteUrl}/search?q={search_term_string}`,
      'query-input': 'required name=search_term_string',
    },
  }
  
  useStructuredData(websiteData)
}

/**
 * Generate FAQ structured data
 */
export function useFAQStructuredData(faqs: Array<{ question: string; answer: string }>) {
  const faqData: StructuredDataConfig = {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: faqs.map(faq => ({
      '@type': 'Question',
      name: faq.question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: faq.answer,
      },
    })),
  }
  
  useStructuredData(faqData)
}
