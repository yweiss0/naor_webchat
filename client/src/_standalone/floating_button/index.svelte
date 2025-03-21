

<script>
    import '../../app.css';
    // import '../../shared/app.css';
    let isChatOpen = false;
    let isLoading = false;
    let messageInput = '';
    let messages = [
        {
            role: 'assistant',
            content: 'How can I help you?'
        }
    ];

    function openChat() {
        isChatOpen = !isChatOpen;
    }
    

    async function handleSubmit(event) {
        event.preventDefault();
        if (messageInput.trim()) {
            isLoading = true; // Start loading
            let userMessage = { role: 'user', content: messageInput.trim() };
            messages = [...messages, userMessage];
            messageInput = ''; // Clear input

            try {
                const response = await fetch('http://localhost:8000/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userMessage.content }),
                });

                const data = await response.json();
                let aiMessage = { role: 'assistant', content: data.response };
                messages = [...messages, aiMessage];
            } catch (error) {
                console.error('Error sending message:', error);
                messages = [...messages, { role: 'assistant', content: "Sorry, something went wrong." }];
            } finally {
                isLoading = false; // Stop loading
            }
        }
    }


    function handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(event);
        }
    }

    async function sendHealthCheck() {
        try {
            const response = await fetch('http://localhost:8000/health', {
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
            // Convert newlines to paragraphs
            .split('\n\n').map(paragraph => 
                `<p>${paragraph.replace(/\n/g, '<br>')}</p>`
            ).join('');

        // Add line breaks for any remaining newlines
        formatted = formatted.replace(/\n/g, '<br>');
        
        return formatted;
    }
</script>

<button class="fixed bottom-5 right-5 rounded-full bg-black text-white p-3" on:click={openChat}>
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
  class="fixed bottom-24 right-0 mr-4 bg-white p-6 rounded-lg border border-[#e5e7eb] w-[440px] h-[450px] lg:h-[550px] flex flex-col z-50"
  style="box-shadow: 0 0 #0000, 0 0 #0000, 0 1px 2px 0 rgb(0 0 0 / 0.05);"
>
    <div class="flex flex-col space-y-1.5 pb-6">
      <h2 class="font-semibold text-lg tracking-tight">Chatbot</h2>
    </div>

    <div class="pr-4 flex-1 overflow-y-auto">
      {#each messages as message}
      {#if message.role === 'user'}
       <!-- User message aligned to left -->
      <div class="flex gap-3 my-4 text-gray-600 text-sm">
          <span class="relative flex shrink-0 overflow-hidden rounded-full w-8 h-8">
              <div class="rounded-full bg-gray-100 border p-1">
                  <svg fill="black" viewBox="0 0 16 16" height="20" width="20">
                      <path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4Zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10Z"></path>
                  </svg>
              </div>
          </span>
          <p class="leading-relaxed bg-gray-100 p-3 rounded-lg">
              <span class="block font-bold text-gray-700">You </span>
              {message.content}
          </p>
      </div>
          {:else}
          <!-- AI message aligned to right -->
          <div class="flex flex-row-reverse gap-3 my-4 text-gray-600 text-sm ml-auto max-w-[80%]">
              <span class="relative flex shrink-0 overflow-hidden rounded-full w-8 h-8">
                  <div class="rounded-full bg-gray-100 border p-1">
                      <svg fill="black" viewBox="0 0 24 24" height="20" width="20">
                          <path d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z"></path>
                      </svg>
                  </div>
              </span>
              <div class="leading-relaxed bg-blue-50 p-3 rounded-lg shadow-sm ml-2 w-full">
                <span class="block font-bold text-gray-700 text-right">AI </span>
                <div class="text-left">
                    {@html formatResponse(message.content)}
                </div>
            </div>
          </div>
          {/if}
      {/each}
    </div>

    <form class="flex items-center pt-0" on:submit={handleSubmit}>
      <div class="flex items-center justify-center w-full space-x-2">
        <input
          class="flex h-10 w-full rounded-md border border-[#e5e7eb] px-3 py-2 text-sm placeholder-[#6b7280] focus:outline-none focus:ring-2 focus:ring-[#9ca3af] text-[#030712] focus-visible:ring-offset-2"
          placeholder="Type your message"
          bind:value={messageInput}
          on:keypress={handleKeyPress}
        >
        <button 
          type="submit"
          class="inline-flex items-center justify-center rounded-md text-sm font-medium text-white bg-black hover:bg-[#111827E6] h-10 px-4 py-2"
        >
          <svg class="w-5 h-5 mr-2" fill="white" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
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

