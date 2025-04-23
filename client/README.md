# naor_webchat

Svelte standalone web app for stock chatbot (use **Option B**).



---

## How to Create a Svelte Standalone App (Option A)
This optin uses  [Svelte Standalone](https://standalone.brenoliradev.com/).
1. **Create a Svelte App**  
   ```bash
   npx sv create my-app   # you can add Tailwind
   cd my-app
   npm install
   ```

2. **Clean up boilerplate**  
   Remove the following files if they exist:  
   - `src/components/Counter.svelte`  
   - `src/assets/svelte.svg`  
   - `src/routes/+page.svelte`  

3. **Install Svelte Standalone**  
   ```bash
   npm install -D svelte-standalone@latest
   ```

4. **Generate a standalone component**  
   ```bash
   npx standalone create
   ```  
   - **Component name:** e.g. `payments`  
   - **Embedding strategy:**  
     - A. Explicit Call (Single Instance): `window.payments.start()`  
     - B. Explicit Call (Multiple Instances): `window.payments.start()`  
     - C. Auto‑Embed with Target ID  
     - D. Auto‑Embed with Target Class  
     - E. Auto‑Embed on Body  

5. **Configure your component**  
   Navigate to `src/_standalone/<componentName>` (e.g. `payments`):  
   - Create `embed.js` with:
     ```js
     import { autoEmbedWithTarget } from 'svelte-standalone';
     import Example from './index.svelte'; // your embeddable component
     autoEmbedWithTarget(Example, 'example');
     ```
   - `index.svelte` is where you write your Svelte code.  
   - To use Tailwind, import your CSS in each component’s `<script>` block:
     ```js
     import '../../app.css';
     ```

6. **Build the project**  
   ```bash
   npx standalone build --all
   ```

7. **Test the output**  
   - After building, find `componentName.min.js` in `client/static/dist/standalone`.  
   - In the same folder, create `index.html`:
     ```html
     <!DOCTYPE html>
     <html>
       <head>
         <title>Floating Button Test</title>
       </head>
       <body>
         <h1>Testing the Floating Button</h1>
         <p>This is some content to test with.</p>
         <script src="tryitnow.min.js"></script>
         <floating-button></floating-button>
       </body>
     </html>
     ```
   - Serve it (for example, `http-server .`) and open `http://localhost:8080/index.html`.

8. **Embed in WordPress**  
   - Add a **Custom HTML** block.  
   - Insert:
     ```html
     <script src="http://localhost:8080/floating_button.min.js"></script>
     ```  
   - Verify it on your site.

---

## Option B – Create a Web Component (**Recommended**)

1. **Initialize a Vite + Svelte project** (NOT SvelteKit):  
   ```bash
   npm create vite my-web-component
   ```  
   Select **Svelte**.

2. **Install dependencies**  
   ```bash
   cd my-web-component
   npm install
   ```

3. **Set up Tailwind CSS (v3)**  
   ```bash
   npm install -D tailwindcss@3 postcss autoprefixer
   npx tailwindcss init -p
   ```  
   - **`tailwind.config.js`**  
     ```js
     /** @type {import('tailwindcss').Config} */
     export default {
       content: ['./index.html', './src/**/*.{svelte,js,ts,jsx,tsx}'],
       theme: { extend: {} },
       plugins: []
     };
     ```
   - **`app.css`**  
     ```css
     @tailwind base;
     @tailwind components;
     @tailwind utilities;
     ```

4. **Configure Svelte as a custom element**  
   - **`svelte.config.js`**  
     ```js
     import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

     export default {
       preprocess: vitePreprocess(),
       compilerOptions: {
         customElement: true
       }
     };
     ```

5. **Configure Vite for library build**  
   - **`vite.config.js`**  
     ```js
     import { defineConfig } from 'vite';
     import { svelte } from '@sveltejs/vite-plugin-svelte';

     export default defineConfig({
       build: {
         lib: {
           name: 'svelteWebComponents',
           entry: 'src/main.js',
           formats: ['iife'],
           fileName: 'swc'
         }
       },
       plugins: [svelte()]
     });
     ```

6. **Create your component file**  
   Place your Svelte file in `/lib` (e.g. `ChatWidget.svelte`).

7. **Declare the custom element**  
   ```svelte
   <svelte:options customElement="swc-chatwidget" />
   ```
   > Must include a hyphen (e.g. `swc-chatwidget`).

8. **Import Tailwind within the component**  
   ```svelte
   <style>
     @import 'tailwindcss/base';
     @import 'tailwindcss/components';
     @import 'tailwindcss/utilities';
   </style>
   ```

9. **Update the entry script**  
   - **`src/main.js`**  
     ```js
     import './app.css';
     import ChatWidget from './lib/ChatWidget.svelte';
     ```

10. **Build the library**  
    ```bash
    npm run build
    ```

11. **Retrieve the bundle**  
    In `dist/`, keep only `swc.iife.js`.

12. **Host the file**  
    Upload `swc.iife.js` to your server or CDN.

13. **Use the component**  
    ```html
    <script type="module" src="https://your-host.com/swc.iife.js"></script>
    <swc-chatwidget></swc-chatwidget>
    ```
    > Ensure the tag name matches `<svelte:options customElement>`.

---

## How to Implement with Google Tag Manager (GTM)

1. Create a GTM account and connect via the GTM4WP plugin in WordPress.  
2. Add a new **Custom HTML** tag and insert:

   <!-- Load your IIFE bundle -->
   ```html
   <script src="https://your_vps_hosting_the_component/swc.iife.js" id="swc-script-loader"></script>
   ```

   <!-- Fallback logic for race conditions -->
   ```html
   <script>
   (function() {
     console.log('[SWC Debug V2] Inline script started.');

     function createAndAppendWidget() {
       if (customElements.get('swc-chatwidget')) {
         console.log('[SWC Debug V2] Definition found. Appending widget...');
         try {
           var chatWidget = document.createElement('swc-chatwidget');
           document.body.appendChild(chatWidget);
           console.log('[SWC Debug V2] Successfully appended swc-chatwidget.');
         } catch (e) {
           console.error('[SWC Debug V2] Error creating/appending element:', e);
         }
       } else {
         console.error('[SWC Debug V2] swc-chatwidget still not defined!');
       }
     }

     if (customElements.get('swc-chatwidget')) {
       console.log('[SWC Debug V2] Element defined. Creating immediately.');
       createAndAppendWidget();
     } else {
       console.log('[SWC Debug V2] Waiting for loader script…');
       var loader = document.getElementById('swc-script-loader');
       if (!loader) {
         console.error('[SWC Debug V2] Could not find loader script tag!');
         return;
       }
       loader.addEventListener('load', function() {
         console.log('[SWC Debug V2] Loader loaded. Attempting creation.');
         createAndAppendWidget();
       });
       loader.addEventListener('error', function(event) {
         console.error('[SWC Debug V2] Error loading swc.iife.js!', event);
       });
       console.log('[SWC Debug V2] Fallback listeners attached.');
     }
   })();
   </script>
   ```

   <!-- Final debug log -->
   ```html
   <script>
     console.log("here from gtm");
   </script>
   ```
