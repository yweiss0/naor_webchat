<script>
    import { Button } from "$lib/components/ui/button";
    import Icon from "@iconify/svelte";

    let isChatOpen = false;
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

    function handleSubmit(event) {
        event.preventDefault();
        if (messageInput.trim()) {
            messages = [...messages, {
                role: 'user',
                content: messageInput.trim()
            }];
            messageInput = '';
        }
    }

    function handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(event);
        }
    }
</script>

<Button class="fixed bottom-5 right-5 rounded-full" on:click={openChat}>
    <Icon class="mr-2 w-6 h-6" icon="fluent:chat-sparkle-20-regular" />
    Ask AI 
</Button>

{#if isChatOpen}
<div 
  class="fixed bottom-24 right-0 mr-4 bg-white p-6 rounded-lg border border-[#e5e7eb] w-[440px] h-[450px] lg:h-[550px] flex flex-col z-50"
  style="box-shadow: 0 0 #0000, 0 0 #0000, 0 1px 2px 0 rgb(0 0 0 / 0.05);"
>
    <!-- Heading -->
    <div class="flex flex-col space-y-1.5 pb-6">
      <h2 class="font-semibold text-lg tracking-tight">Chatbot</h2>
    </div>

    <!-- Chat Container -->
    <div class="pr-4 flex-1 overflow-y-auto">
      {#each messages as message}
        {#if message.role === 'user'}
          <div class="flex gap-3 my-4 text-gray-600 text-sm">
            <span class="relative flex shrink-0 overflow-hidden rounded-full w-8 h-8">
              <div class="rounded-full bg-gray-100 border p-1">
                <svg 
                  stroke="none" 
                  fill="black" 
                  stroke-width="0"
                  viewBox="0 0 16 16" 
                  height="20" 
                  width="20" 
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4Zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10Z"></path>
                </svg>
              </div>
            </span>
            <p class="leading-relaxed">
              <span class="block font-bold text-gray-700">You </span>
              {message.content}
            </p>
          </div>
        {:else}
          <div class="flex flex-row-reverse gap-3 my-4 text-gray-600 text-sm">
            <span class="relative flex shrink-0 overflow-hidden rounded-full w-8 h-8">
              <div class="rounded-full bg-gray-100 border p-1">
                <svg 
                  stroke="none" 
                  fill="black" 
                  stroke-width="1.5"
                  viewBox="0 0 24 24" 
                  height="20" 
                  width="20" 
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path 
                    stroke-linecap="round" 
                    stroke-linejoin="round"
                    d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
                  ></path>
                </svg>
              </div>
            </span>
            <p class="leading-relaxed">
              <span class="block font-bold text-gray-700">AI </span>
              {message.content}
            </p>
          </div>
        {/if}
      {/each}
    </div>

    <!-- Input box -->
    <form class="flex items-center pt-0" on:submit={handleSubmit}>
      <div class="flex items-center justify-center w-full space-x-2">
        <input
          class="flex h-10 w-full rounded-md border border-[#e5e7eb] px-3 py-2 text-sm placeholder-[#6b7280] focus:outline-none focus:ring-2 focus:ring-[#9ca3af] text-[#030712] focus-visible:ring-offset-2"
          placeholder="Type your message"
          bind:value={messageInput}
          on:keypress={handleKeyPress}
        >
        <Button 
          type="submit"
          class="inline-flex items-center justify-center rounded-md text-sm font-medium text-[#f9fafb] bg-black hover:bg-[#111827E6] h-10 px-4 py-2"
        >
          <Icon class="mr-2 w-5 h-5" icon="fluent:send-20-filled" />
          Send
        </Button>
      </div>
    </form>
  </div>
{/if}

<style>
    @media (max-width: 640px) {
        .fixed.bottom-4.right-4 {
            width: 90vw !important;
            height: 70vh !important;
            max-height: 70vh;
            max-width: 90vw;
            margin-right: 1rem;
        }
        .fixed.bottom-24.right-0 {
            width: 90vw !important;
            height: 70vh !important;
            max-height: 70vh;
            max-width: 90vw;
            margin-right: 1rem;
        }
    }
</style>