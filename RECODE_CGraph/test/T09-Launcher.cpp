//这是我写的一个简单的教程，用于启动OneDragon-Launcher、BetterGI、BetterMD2等游戏自动化程序，用于测试CGraph的功能。
#include "../src/GraphCtrl/GraphInclude.h"
#include <cstdlib>
#include <chrono>
#include <thread>
class OneDragonNode : public GNode {
public:
    CSTATUS run() override {
        CSTATUS status = STATUS_OK;
        CGRAPH_ECHO("[%s], start OneDragon-Launcher ... ", this->getName().c_str());
        std::string command = "start /wait \"\" \"D:\\BETTERZZZ\\OneDragon-Launcher.exe\" -o -c";
        int result = system(command.c_str());
        if (result != 0) {
            CGRAPH_ECHO("[%s], OneDragon-Launcher execution failed with code: %d", this->getName().c_str(), result);
        }
        CGRAPH_ECHO("[%s], OneDragon-Launcher execution succeeded", this->getName().c_str());
        return status;
    }
};

class BetterGINode : public GNode {
public:
    CSTATUS run() override {
        CSTATUS status = STATUS_OK;
        CGRAPH_ECHO("[%s], start BetterGI ... ", this->getName().c_str());
        std::string command = "\"D:\\BETTERGI\\BetterGI.exe\" -startOneDragon";
        int result = system(command.c_str());
        if (result != 0) {
            CGRAPH_ECHO("[%s], BetterGI execution failed with code: %d", this->getName().c_str(), result);
        }
        CGRAPH_ECHO("[%s], BetterGI execution succeeded", this->getName().c_str());
        return status;
    }
};
class BetterMD2Node : public GNode {
public:
    CSTATUS run() override {
        CSTATUS status = STATUS_OK;

        // 等待至北京时间早上8:00
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm;
        localtime_s(&local_tm, &now_c);

        if (local_tm.tm_hour < 8) {
            int seconds_to_wait = (8 - local_tm.tm_hour) * 3600
                                  - local_tm.tm_min * 60
                                  - local_tm.tm_sec;
            CGRAPH_ECHO("[%s], 等待 %d 秒至北京时间 8:00 ...", this->getName().c_str(), seconds_to_wait);
            std::this_thread::sleep_for(std::chrono::seconds(seconds_to_wait));
        }

        CGRAPH_ECHO("[%s], start BetterMD2 ... ", this->getName().c_str());
        std::string command = "\"D:\\BETTERMD2\\MFABD2-v4.3.10-win-x86_64\\MFAAvalonia.exe";
        int result = system(command.c_str());
        if (result != 0) {
            CGRAPH_ECHO("[%s], BetterMD2 execution failed with code: %d", this->getName().c_str(), result);
        }
        CGRAPH_ECHO("[%s], BetterMD2 execution succeeded", this->getName().c_str());
        return status;
    }
};
int tutorial_launcher() {
    GPipelinePtr pipeline = GPipelineFactory::create();
    CSTATUS status = STATUS_OK;
    GElementPtr one_dragon, better_gi = nullptr, better_md2 = nullptr;

    status = pipeline->registerGElement<OneDragonNode>(&one_dragon, {}, "OneDragon");
    if (STATUS_OK != status) {
        return status;
    }

    status = pipeline->registerGElement<BetterGINode>(&better_gi, {one_dragon}, "BetterGI");
    if (STATUS_OK != status) {
        return status;
    }
    status = pipeline->registerGElement<BetterMD2Node>(&better_md2, {one_dragon}, "BetterMD2");
    if (STATUS_OK != status) {
        return status;
    }
        
    status = pipeline->init();
    if (STATUS_OK != status) {
        return status;
    }

    status = pipeline->run();
    CGRAPH_ECHO("tutorial_launcher run status = %d", status);

    status = pipeline->deinit();
    GPipelineFactory::destroy(pipeline);
    return status;
}

int main() {
    int status = tutorial_launcher();
    std::cout << status;
    return 0;
}
