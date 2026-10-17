/**
 * Composable for loading module content from Nuxt Content
 * 
 * Usage:
 * const { content } = await useModuleContent('classification', 'logistic-regression')
 */
export const useModuleContent = async (domain: string, moduleId: string) => {
  const { data: content } = await useAsyncData(
    `${domain}-${moduleId}-content`,
    () => queryCollection('content').path(`/modules/${domain}/${moduleId}/content`).first()
  );

  return {
    content
  };
};
