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
    SDL_Window* window;

	VkInstance instance;
    

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
        window = SDL_CreateWindow("Vulkan Triangle with SDL3", WIDTH, HEIGHT,
            SDL_WINDOW_VULKAN); // Remove the flag SDL_WINDOW_RESIZABLE

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
        // --- Get required Vulkan extensions from SDL3 ---
        Uint32 extensionCount = 0;
        const char* const* SDL3_Extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
        if (!SDL3_Extensions || extensionCount == 0) {
            std::cerr << "Failed to get Vulkan extension count: " << SDL_GetError() << '\n';
            SDL_DestroyWindow(window);
            SDL_Quit();
            throw std::runtime_error("Failed to get Vulkan extensions!");
		}

        #ifndef NDEBUG
        std::cout << "Number of Vulkan extensions: " << extensionCount << '\n';
        for (Uint32 i = 0; i < extensionCount; ++i) {
            std::cout << "Extension " << i << ": " << SDL3_Extensions[i] << '\n';
		}
        #endif // !NDEBUG
        // --- Create Vulkan Instance ---
        VkApplicationInfo appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "SDL3 Vulkan Demo",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_4
        };
        VkInstanceCreateInfo createInfo = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = extensionCount,
            .ppEnabledExtensionNames = SDL3_Extensions
        };
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance\n";
            SDL_DestroyWindow(window);
            SDL_Quit();
            throw std::runtime_error("Failed to create Vulkan instance!");
        }
        std::cout << "Vulkan instance created successfully!\n";
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


        vkDestroyInstance(instance, nullptr);

        SDL_DestroyWindow(window);
        SDL_Quit();
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





/*
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>


const int width = 800;
const int height = 600;

int main() {
    // --- Initialize SDL3 ---
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << '\n';
        std::cerr << "This might indicate SDL3.dll is missing or corrupted.\n";
        std::cerr << "Please ensure SDL3.dll is in the same directory as the executable.\n";
        return 1;
    }

    std::cout << "SDL3 initialized successfully!\n";

    // --- Create SDL3 Window ---
    SDL_Window* window = SDL_CreateWindow("Vulkan Triangle with SDL3", width, height,
        SDL_WINDOW_VULKAN); // Remove the flag SDL_WINDOW_RESIZABLE

    if (window == nullptr) {
        std::cerr << "Window creation failed: " << SDL_GetError() << '\n';
        SDL_Quit();
        return 1;
    }

    std::cout << "SDL3 window created successfully!\n";

    if (!SDL_Vulkan_LoadLibrary(nullptr)) {
        std::cerr << "Failed to load Vulkan library: " << SDL_GetError() << '\n';
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "Vulkan library loaded successfully!\n";

    // --- Get required Vulkan extensions from SDL3 ---
    Uint32 extensionCount = 0;
    const char* const* SDL3_Extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
    if (!SDL3_Extensions || extensionCount == 0) {
        std::cerr << "Failed to get Vulkan extension count: " << SDL_GetError() << '\n';
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

#ifndef NDEBUG
    std::cout << "Number of Vulkan extensions: " << extensionCount << '\n';
    for (Uint32 i = 0; i < extensionCount; ++i) {
        std::cout << "Extension " << i << ": " << SDL3_Extensions[i] << '\n';
    }
#endif // !NDEBUG

    // --- Create Vulkan Instance ---
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "SDL3 Vulkan Demo",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3
    };

    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = extensionCount,
        .ppEnabledExtensionNames = SDL3_Extensions
    };

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "Vulkan instance created successfully!\n";

    // --- Create Vulkan Surface ---
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        std::cerr << "Surface creation failed: " << SDL_GetError() << '\n';
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::cout << "SDL3 Vulkan surface created successfully!\n";

    // --- Main Loop ---
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

    // --- Cleanup ---
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
*/