# naor_webchat
Svelte standalone web app for stock chatbot.

this app made with Svelte Standalone - https://standalone.brenoliradev.com/

How to create a Svelte standalone app?

1.Create a Svelte App:
    a. npx sv create my-app (you can add tailwind)
    b. cd my-app
    c. npm install

2. Clean Up – Remove svelte boilerplate code (e.g., routes, components).
    Files to delete:
    a. src/components/Counter.svelte (if exist)
    b. src/assets/svelte.svg (if exist)
    c. src/routes/+page.svelte (if exists)
3. Install Svelte Standalone
    a. npm install -D svelte-standalone@latest

4. Generates boilerplate code for a new standalone component
    a. npx standalone create
    b. Name your component: Enter a name for your component (e.g., payments).
    c. Choose an embedding strategy (for the chatbot option B is teh best):
        A. Explicit Call (Single Instance): Mounts the component once using window.payments.start().
        B. Explicit Call (Multiple Instances): Allows mounting multiple instances with window.payments.start().
        C. Auto-Embed with Target ID: Automatically appends to an element with a specified id.
        D. Auto-Embed with Target Class: Automatically appends to elements with a specified class.
        E. Auto-Embed on Body: Automatically appends to the <body> when downloaded.
5. now look at src/_standalone/name_of_your_component (e.g payments)
    a. create file and name it embed.js and put this code in it:
        import { autoEmbedWithTarget } from 'svelte-standalone';
        import Example from './index.svelte'; // your embedabble
        autoEmbedWithTarget(Example, 'example');
    b. index.svelte - this is the file where you code your regular svelte code
    c. for using tailwind you need to import tailwind in each index.svelte component jsut add in the <script> import '../../app.css';
6.  Build the app
    a.npx standalone build --all
7. test the app
    a. step 6 will create a files in client\static\dist\standalone\componentName.min.js
    b create a file call it index.html and put this code in it:
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

    c. serve the file with http-server from the folder client\static\dist\standalone
    d. open localhost:8080/index.html and you will see your file
8. add the svelte component to WordPress
    a. add new block in wordpress and select 'custom html'
    b. add there this code:
        <script src="http://localhost:8080/floating_button.min.js"></script>
    c. test the WordPress website.

