#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    SDL_Window* window =nullptr;

	VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    void initWindow() {
        // --- Initialize SDL3 ---
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << '\n';
            std::cerr << "This might indicate SDL3.dll is missing or corrupted.\n";
            std::cerr << "Please ensure SDL3.dll is in the same directory as the executable.\n";
            throw std::runtime_error("Failed initialized SDL!");
        }

        std::cout << "SDL3 initialized successfully!\n";

        // --- Create SDL3 window ---
        window = SDL_CreateWindow("Vulkan Triangle with SDL3", WIDTH, HEIGHT, SDL_WINDOW_VULKAN); // Remove the flag SDL_WINDOW_RESIZABLE

        if (window == nullptr) {
            std::cerr << "Window creation failed: " << SDL_GetError() << '\n';
            SDL_Quit();
            throw std::runtime_error("failed SDL3 window!");
        }

        std::cout << "SDL3 window created successfully!\n";

        if (!SDL_Vulkan_LoadLibrary(nullptr)) {
            std::cerr << "Failed to load Vulkan library: " << SDL_GetError() << '\n';
            SDL_DestroyWindow(window);
            SDL_Quit();
            throw std::runtime_error("Failed Load SDL3 Vulkan library!");
        }

        std::cout << "SDL3 Vulkan library loaded successfully!\n";
    }

    void initVulkan() {
        createInstance(); 
        createSurface();
    }

    void mainLoop() {
        bool running = true;
        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT) {
                    running = false;
                }
            }

            // Add rendering code here later
        }
    }

    void cleanup() {

		vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    void createInstance() {
        // --- Get required Vulkan extensions from SDL3 ---
        
		auto extensions = getRequiredExtensions();

        // --- Create Vulkan Instance ---
        VkApplicationInfo appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "SDL3 Vulkan Demo",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_4, // Use the latest Vulkan API version supported by your SDK
        };
        // --- Create Vulkan Instance Create Info ---
        VkInstanceCreateInfo createInfo = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext = nullptr,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(), 
        };

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance!");
        }
        std::cout << "Vulkan instance created successfully!\n";
    }

    void createSurface() {
        // --- Create Vulkan Surface ---
        if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
            std::cerr << "Surface creation failed: " << SDL_GetError() << '\n';
            vkDestroyInstance(instance, nullptr);
            SDL_DestroyWindow(window);
            SDL_Quit();
            throw std::runtime_error("Failed to create Vulkan surface!");
        }
        std::cout << "Vulkan surface created by SDL3 successfully!\n";
    }

    std::vector<const char*> getRequiredExtensions() {
        Uint32 extensionCount = 0;
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        if (!extensions || extensionCount == 0) {
            throw std::runtime_error("Failed to get Vulkan extensions from SDL3!");
        }
        std::vector<const char*> requiredExtensions(extensions, extensions + extensionCount);
	
#ifndef NDEBUG
		std::cout << "Required Vulkan extensions:\n";
        for (const char* ext : requiredExtensions) {
            std::cout << "  " << ext << '\n';
		}
#endif !NDEBUG

        return requiredExtensions;
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
