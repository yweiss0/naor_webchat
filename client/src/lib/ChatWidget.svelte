<svelte:options customElement="swc-chatwidget" />
<script>
  // import '../app.css';
  import { afterUpdate, tick } from 'svelte';

  let isChatOpen = false;
  let isLoading = false;
  let messageInput = '';
  let typingMessageId = null;
  let messages = [];
  let messagesContainer;
  let userScrolledUp = false;
  let chatInitialized = false; // Track if the initial AI message has been sent


  function openChat() {
      isChatOpen = !isChatOpen;

      if (isChatOpen && !chatInitialized) {
          // Send initial AI message if chat is opened for the first time
          sendInitialMessage();
          chatInitialized = true; // Mark as initialized
      }
  }
  function closeChat() {
  isChatOpen = false;
  }
  

  async function typeMessage(content, messageId) {
      const message = messages.find(m => m.id === messageId);
      let index = 0;
      
      while (index < content.length) {
          message.content = content.slice(0, index + 1);
          messages = messages;
          index++;
          await new Promise(resolve => setTimeout(resolve, 20));
           // Scroll while typing
          if(messagesContainer) scrollToBottom(messagesContainer);
      }
  }

  async function handleSubmit(event) {
      event.preventDefault();
      if (messageInput.trim() && !isLoading) {
          isLoading = true;
          const userMessage = {
              id: Date.now(),
              role: 'user',
              content: messageInput.trim()
          };

          // Add temporary thinking message
          const thinkingMessage = {
              id: Date.now() + 1,
              role: 'assistant',
              content: 'Thinking...',
              status: 'thinking'
          };

          messages = [...messages, userMessage, thinkingMessage];
          messageInput = '';

          // Scroll after adding user message and thinking message
          if(messagesContainer) scrollToBottom(messagesContainer);

          try {
              const response = await fetch('https://nextdawnai.cloud/api/chat', { // Your backend URL
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ message: userMessage.content }),
                  credentials: 'include' // *** Ensure this line is present ***
              });

              // Check for HTTP errors first
              if (!response.ok) {
                  let errorDetail = `HTTP error ${response.status}`;
                  try {
                      // Try to get more specific error from backend response body
                      const errorData = await response.json();
                      errorDetail = errorData.detail || errorDetail;
                  } catch (e) {
                      // Ignore if response body isn't JSON or empty
                      errorDetail = `${errorDetail}: ${response.statusText}`;
                  }
                  throw new Error(errorDetail); // Throw an error to be caught below
              }

              const data = await response.json();
              const aiMessage = {
                  id: Date.now() + 2, // Ensure unique ID
                  role: 'assistant',
                  content: '',
                  status: 'typing'
              };

              // Replace thinking message with typing message
              messages = messages.filter(m => m.id !== thinkingMessage.id);
              messages = [...messages, aiMessage];
              typingMessageId = aiMessage.id;

              // Start typing animation
              await typeMessage(data.response, aiMessage.id);

          } catch (error) {
              // Catch both network errors and HTTP errors thrown above
              console.error('Error submitting message:', error);
              messages = messages.filter(m => m.id !== thinkingMessage.id); // Remove thinking message
              messages = [...messages, {
                  id: Date.now() + 3, // Unique ID
                  role: 'assistant',
                  content: `Sorry, there was an error: ${error.message}` // Display error
              }];
               // Scroll after error
              if(messagesContainer) scrollToBottom(messagesContainer);
          } finally {
              isLoading = false;
              typingMessageId = null;
              // Scroll after AI message is complete (or error)
              if(messagesContainer) scrollToBottom(messagesContainer);
          }
      }
  }

      async function sendInitialMessage() {
      const aiMessage = {
          id: Date.now(),
          role: 'assistant',
          content: '',
          status: 'typing'
      };
      messages = [...messages, aiMessage];
      typingMessageId = aiMessage.id;
      await typeMessage("Hi there! I'm Nixi, your trading assistant.\n Need market insights, stock updates, or strategy tips? I've got you covered! What's on your mind today? \n\n The information provided by this AI bot is for general informational and educational purposes only. It does not constitute investment advice, a personal recommendation, or an offer or solicitation to buy or sell any financial instruments. Please refer to the Terms & Conditions <a href='https://www.nrdx.com/wp-content/uploads/NRDX_TCs_final.pdf'>https://www.nrdx.com/terms&conditions</a> available on our website.", aiMessage.id);}

  function handleKeyPress(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          handleSubmit(event);
      }
  }

  async function sendHealthCheck() {
      //  a function to test the api connectivity
      try {
          const response = await fetch('https://nextdawnai.cloud/api/health', {
              method: 'GET'
          });
          const data = await response.json();
          console.log(data);
          messages = [...messages, {
              role: 'user',
              content: messageInput.trim()
          }];
          messageInput = '';
      } catch (error) {
          console.error('Error:', error);
      }
  }
  function formatResponse(text) {
      // Convert Markdown-like syntax to HTML
      let formatted = text
          // Convert bold (**text**)
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          // Convert italic (*text*)
          .replace(/\*(.*?)\*/g, '<em>$1</em>')
          // Style links with same color as text (#CCFF00) for consistency
          .replace(/<a href="(https?:\/\/[^"]+)"([^>]*)>(.*?)<\/a>/g, '<a href="$1"$2 style="color: #CCFF00; text-decoration: underline;">$3</a>')
          // Style email links with same color as text (#CCFF00) for consistency
          .replace(/<a href="mailto:([^"]+)"([^>]*)>(.*?)<\/a>/g, '<a href="mailto:$1"$2 style="color: #CCFF00; text-decoration: underline;">$3</a>')
          // Convert newlines to paragraphs
          .split('\n\n').map(paragraph => 
              `<p>${paragraph.replace(/\n/g, '<br>')}</p>`
          ).join('');

      // Add line breaks for any remaining newlines
      formatted = formatted.replace(/\n/g, '<br>');
      
      return formatted;
  }


  const scrollToBottom = async (node) => {
  if (node) {
      node.scroll({ top: node.scrollHeight, behavior: 'smooth' });
  }
  };

  afterUpdate(() => {
  if (messagesContainer) {
      scrollToBottom(messagesContainer);
  }
  });

  // Track scroll events
  function handleScroll() {
      const fromBottom = messagesContainer.scrollHeight - 
                      messagesContainer.scrollTop - 
                      messagesContainer.clientHeight;
      userScrolledUp = fromBottom > 100;
  }
</script>
<button class="fixed bottom-24 right-5 z-40 rounded-full bg-black text-white p-3" on:click={openChat}>
  <svg class="w-6 h-6 inline-block" fill="white" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 3C6.477 3 2 7.03 2 12c0 2.064.838 3.976 2.238 5.477L4 21l3.596-1.798A9.959 9.959 0 0012 21c5.523 0 10-4.03 10-9s-4.477-9-10-9z"></path>
  </svg>
  Ask AI 
</button>
<!-- <button class="fixed bottom-16 right-5 rounded-full bg-black text-white p-4" on:click={sendHealthCheck}>
  Health Check
</button> -->
{#if isChatOpen}
  <div 
  class="fixed bottom-24 right-0 mr-4 bg-white p-6 rounded-lg border border-[#e5e7eb] w-[400px] h-[400px] lg:h-[550px] flex flex-col z-50"
  style="box-shadow: 0 0 #0000, 0 0 #0000, 0 1px 2px 0 rgb(0 0 0 / 0.05);"
  >
      <div class="flex flex-col space-y-1.5 pb-6">
      <h2 class="font-semibold text-lg tracking-tight text-black">NRDX AI Bot</h2>
      </div>
      <button 
          class="absolute top-2 right-2 text-gray-500 hover:text-red-500 text-xl"
          on:click={closeChat}
      >
          &times;
      </button>
      <div class="pr-4 flex-1 overflow-y-auto" bind:this={messagesContainer} on:scroll={handleScroll}>
      {#each messages as message}
          {#if message.role === 'user'}
          <!-- User message aligned to left -->
          <div class="flex gap-3 my-4 text-[#CCFF00] text-sm">
              <span class="relative flex shrink-0 overflow-hidden rounded-full w-8 h-8">
                  <div class="rounded-full bg-[#CCFF00] border p-1">
                      <svg fill="black" viewBox="0 0 16 16" height="20" width="20">
                          <path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4Zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10Z"></path>
                      </svg>
                  </div>
              </span>
              <p class="leading-relaxed bg-black p-3 rounded-lg">
                  <span class="block font-bold text-white">You </span>
                  {message.content}
              </p>
          </div>
          {:else}
                  <div class="flex flex-row-reverse gap-3 my-4 text-[#CCFF00] text-sm ml-auto max-w-[80%]">
                      <span class="relative flex shrink-0 overflow-hidden rounded-full w-8 h-8">
                          {#if message.status === 'thinking'}
                              <!-- Spinner -->
                              <div class="rounded-full bg-[#202020] border  p-1">
                                  <svg class="animate-spin size-5 text-[#CCFF00]" viewBox="0 0 24 24">
                                      <path fill="currentColor" d="M12,1A11,11,0,1,0,23,12,11,11,0,0,0,12,1Zm0,19a8,8,0,1,1,8-8A8,8,0,0,1,12,20Z" opacity="0.25"/>
                                      <path fill="currentColor" d="M12,4a8,8,0,0,1,7.89,6.7A1.53,1.53,0,0,0,21.38,12h0a1.5,1.5,0,0,0,1.48-1.75,11,11,0,0,0-21.72,0A1.5,1.5,0,0,0,2.62,12h0a1.53,1.53,0,0,0,1.49-1.3A8,8,0,0,1,12,4Z"/>
                                  </svg>
                              </div>
                          {:else}
                              <!-- AI Icon -->
                              <div class="rounded-full bg-[#202020] border border-[#202020] p-1">
                                  <!-- <svg fill="black" viewBox="0 0 24 24" height="20" width="20">
                                      <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z"/>
                                  </svg> -->
                                  <img src="https://nextdawnai.cloud/assets/logo2.bmp" >
                              </div>
                          {/if}
                      </span>
<div class="leading-relaxed bg-black p-3 rounded-lg shadow-sm ml-2 w-full">
              <span class="block font-bold text-white text-right">Nixi </span>
              {#if message.status === 'thinking'}
                  <div class="text-[#CCFF00] italic">
                      {message.content}
                  </div>
              {:else}
                  <div class="text-left text-[#CCFF00] overflow-wrap-anywhere">
                      {@html formatResponse(message.content)}
                  </div>
              {/if}
                  </div>
      </div>
  {/if}
  {/each}
</div>
<form class="flex items-center pt-0" on:submit={handleSubmit}>
    <!-- text input component -->
<div class="flex items-center justify-center w-full space-x-2">
    <input
    class="flex bg-white h-10 w-full rounded-md border border-[#e5e7eb] px-3 py-2 text-sm placeholder-[#6b7280] focus:outline-none focus:ring-2 focus:ring-[#9ca3af] text-[#030712] focus-visible:ring-offset-2"
    placeholder="Type your message"
    bind:value={messageInput}
    on:keypress={handleKeyPress}
    >
    <button 
    type="submit"
    class="inline-flex items-center justify-center rounded-md text-sm font-medium text-[#CCFF00] bg-[#202020] hover:bg-gray-800 h-10 px-4 py-2"
    >
    <svg class="w-5 h-5 mr-2" fill="#CCFF00" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"></path>
    </svg>
    Send
    </button>
    <!-- -- <button class="fixed bottom-16 right-5 rounded-full bg-black text-white p-4" on:click={sendHealthCheck}>
        Health Check
    </button> -->
</div>
</form>
</div>
{/if}
<style>
  @import 'tailwindcss/base';
  @import 'tailwindcss/components';
  @import 'tailwindcss/utilities';
  
  /* Add custom style for text wrapping */
  .overflow-wrap-anywhere {
    overflow-wrap: anywhere;
    word-break: break-word;
  }
</style>