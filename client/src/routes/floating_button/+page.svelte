<script>
	import { onMount } from 'svelte';
  
	onMount(() => {
	  const script = document.createElement('script');
	  script.src = '/src/_standalone/floating_button/embed.js';
	  script.async = true;
	  script.type = 'module';
  
	  document.head.appendChild(script);
  
	  return () => {
		document.head.removeChild(script);
		window.floating_button.stop();
	  };
	});
</script>