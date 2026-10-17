<template>
  <div class="content-section">
    <div class="content-header">
      <h2 class="section-title">{{ title }}</h2>
      <button
        v-if="collapsible"
        @click="toggleCollapse"
        :aria-expanded="!isCollapsed"
        :aria-controls="`content-${sectionId}`"
        class="collapse-button"
      >
        <span class="collapse-icon" :class="{ collapsed: isCollapsed }">
          ▼
        </span>
      </button>
    </div>

    <div
      v-show="!isCollapsed"
      :id="`content-${sectionId}`"
      class="content-body"
    >
      <!-- Markdown content -->
      <div
        v-if="markdownContent"
        class="markdown-content"
        v-html="renderedMarkdown"
      ></div>

      <!-- Slot for custom content -->
      <slot v-else />

      <!-- Images and diagrams -->
      <div v-if="images && images.length > 0" class="content-images">
        <figure
          v-for="(image, index) in images"
          :key="index"
          class="content-figure"
        >
          <img
            :src="image.src"
            :alt="image.alt"
            :width="image.width"
            :height="image.height"
            class="content-image"
            loading="lazy"
          />
          <figcaption v-if="image.caption" class="image-caption">
            {{ image.caption }}
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';

interface ImageData {
  src: string;
  alt: string;
  caption?: string;
  width?: number;
  height?: number;
}

interface Props {
  title?: string;
  markdownContent?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  images?: ImageData[];
}

const props = withDefaults(defineProps<Props>(), {
  collapsible: false,
  defaultCollapsed: false,
  images: () => [],
});

const isCollapsed = ref(props.defaultCollapsed);
const sectionId = ref(`section-${Math.random().toString(36).substr(2, 9)}`);

/**
 * Toggle collapse state
 */
const toggleCollapse = () => {
  isCollapsed.value = !isCollapsed.value;
};

/**
 * Simple markdown renderer
 * This is a basic implementation - for production, use markdown-it or similar
 */
const renderedMarkdown = computed(() => {
  if (!props.markdownContent) return '';

  let html = props.markdownContent;

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Bold
  html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  html = html.replace(/__(.*?)__/gim, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');
  html = html.replace(/_(.*?)_/gim, '<em>$1</em>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // Code blocks
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/gim, (match, lang, code) => {
    return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');

  // Lists
  html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
  html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Paragraphs
  html = html.split('\n\n').map(para => {
    if (para.startsWith('<') || para.trim() === '') return para;
    return `<p>${para}</p>`;
  }).join('\n');

  return html;
});

/**
 * Escape HTML to prevent XSS
 */
const escapeHtml = (text: string): string => {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
};
</script>

<style scoped>
.content-section {
  background: white;
  border-radius: 0.5rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #111827;
  margin: 0;
}

.collapse-button {
  background: none;
  border: none;
  padding: 0.5rem;
  cursor: pointer;
  color: #6b7280;
  transition: color 0.2s;
}

.collapse-button:hover {
  color: #111827;
}

.collapse-button:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
  border-radius: 0.25rem;
}

.collapse-icon {
  display: inline-block;
  transition: transform 0.2s;
  font-size: 0.875rem;
}

.collapse-icon.collapsed {
  transform: rotate(-90deg);
}

.content-body {
  animation: fadeIn 0.2s ease-in;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

/* Markdown Content Styles */
.markdown-content {
  color: #374151;
  line-height: 1.75;
}

.markdown-content :deep(h1) {
  font-size: 1.875rem;
  font-weight: 700;
  color: #111827;
  margin: 1.5rem 0 1rem 0;
}

.markdown-content :deep(h2) {
  font-size: 1.5rem;
  font-weight: 600;
  color: #111827;
  margin: 1.25rem 0 0.75rem 0;
}

.markdown-content :deep(h3) {
  font-size: 1.25rem;
  font-weight: 600;
  color: #111827;
  margin: 1rem 0 0.5rem 0;
}

.markdown-content :deep(p) {
  margin: 0.75rem 0;
}

.markdown-content :deep(strong) {
  font-weight: 600;
  color: #111827;
}

.markdown-content :deep(em) {
  font-style: italic;
}

.markdown-content :deep(a) {
  color: #3b82f6;
  text-decoration: underline;
  transition: color 0.2s;
}

.markdown-content :deep(a:hover) {
  color: #2563eb;
}

.markdown-content :deep(a:focus) {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
  border-radius: 0.125rem;
}

.markdown-content :deep(code) {
  background-color: #f3f4f6;
  color: #dc2626;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  font-size: 0.875em;
  font-family: 'Courier New', Courier, monospace;
}

.markdown-content :deep(pre) {
  background-color: #1f2937;
  color: #f9fafb;
  padding: 1rem;
  border-radius: 0.5rem;
  overflow-x: auto;
  margin: 1rem 0;
}

.markdown-content :deep(pre code) {
  background: none;
  color: inherit;
  padding: 0;
  font-size: 0.875rem;
  line-height: 1.5;
}

.markdown-content :deep(ul),
.markdown-content :deep(ol) {
  margin: 0.75rem 0;
  padding-left: 1.5rem;
}

.markdown-content :deep(li) {
  margin: 0.25rem 0;
}

.markdown-content :deep(ul) {
  list-style-type: disc;
}

.markdown-content :deep(ol) {
  list-style-type: decimal;
}

/* Images */
.content-images {
  margin-top: 1.5rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.content-figure {
  margin: 0;
}

.content-image {
  width: 100%;
  height: auto;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.image-caption {
  margin-top: 0.5rem;
  font-size: 0.875rem;
  color: #6b7280;
  text-align: center;
}

/* Responsive Design */
@media (max-width: 768px) {
  .content-section {
    padding: 1rem;
  }

  .section-title {
    font-size: 1.25rem;
  }

  .markdown-content :deep(h1) {
    font-size: 1.5rem;
  }

  .markdown-content :deep(h2) {
    font-size: 1.25rem;
  }

  .markdown-content :deep(h3) {
    font-size: 1.125rem;
  }

  .content-images {
    grid-template-columns: 1fr;
  }
}
</style>
