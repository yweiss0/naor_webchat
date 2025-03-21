import { autoEmbedWithTarget } from 'svelte-standalone';
        import Example from './index.svelte'; // your embedabble
        autoEmbedWithTarget(Example, 'example');