<template>
  <div v-if="hasError" class="visualization-error">
    <p class="visualization-error-message">{{ errorMessage }}</p>
    <button class="visualization-error-retry" @click="retry">Try Again</button>
  </div>
  <slot v-else />
</template>

<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue';

const hasError = ref(false);
const errorMessage = ref('');

const emit = defineEmits<{
  error: [error: Error];
  retry: [];
}>();

onErrorCaptured((error: Error) => {
  hasError.value = true;
  errorMessage.value = error.message || 'An unexpected error occurred';
  emit('error', error);
  // Prevent error from propagating
  return false;
});

const retry = () => {
  hasError.value = false;
  errorMessage.value = '';
  emit('retry');
};
</script>

<style scoped>
.visualization-error {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  padding: 1.5rem;
  text-align: center;
}

.visualization-error-message {
  color: #6b7280;
  font-size: 0.875rem;
  margin-bottom: 1rem;
}

.visualization-error-retry {
  padding: 0.5rem 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.375rem;
  background: white;
  cursor: pointer;
}
</style>
