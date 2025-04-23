# Naor Chatbot

A comprehensive assistant chatbot with real-time stock market data and AI-powered responses.

## Repository Structure

This repository contains both frontend and backend components of the chatbot:

- **`/client`**: Frontend web component built with Svelte
- **`/server`**: Backend API built with FastAPI

Each directory contains its own detailed README with specific setup and implementation instructions.

## Technology Stack

### Frontend (Web Component)

The frontend is implemented as a standalone web component that can be embedded in any website:

- **Svelte**: Core framework for building the UI
- **Vite**: Build system
- **Tailwind CSS**: Styling
- **Web Components**: Packaged as a custom element for seamless embedding
- **IIFE Bundle**: Single JavaScript file that can be loaded with a script tag

The web component approach allows for easy integration with various platforms, including WordPress and other content management systems, through Google Tag Manager or direct embedding.

### Backend (API Server)

The backend provides intelligent responses and real-time financial data:

- **FastAPI**: API framework for handling requests
- **OpenAI**: GPT models for AI-powered responses
- **Redis**: Session management and conversation history
- **Yahoo Finance API**: Real-time stock and financial data
- **DuckDuckGo Search**: Web search integration for current information
- **Langfuse**: Monitoring and tracing of AI operations

## Quick Start

For detailed setup instructions, refer to the README files in the respective directories:

- [Frontend Setup](/client/README.md)
- [Backend Setup](/server/README.md)

## Deployment

The application is designed for flexible deployment:

- Frontend web component can be hosted on any static file server or CDN
- Backend requires a Python environment with Redis
- Both components can be deployed independently

## Integration

The chatbot can be integrated into any website by including the JavaScript bundle and inserting a custom HTML tag:

```html
<script src="https://your-host.com/swc.iife.js"></script>
<swc-chatwidget></swc-chatwidget>
```

For WordPress sites, Google Tag Manager integration is recommended for easier management.

# How to Implement with Google Tag Manager (GTM)

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
