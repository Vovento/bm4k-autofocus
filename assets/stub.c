
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
static pid_t child = 0;
static void forward(int sig) { if (child > 0) kill(-child, sig); }
int main(void) {
    char path[4096]; uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) return 1;
    char script[4200]; snprintf(script, sizeof(script), "%s.sh", path);
    posix_spawnattr_t attr; posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP); posix_spawnattr_setpgroup(&attr, 0);
    char *args[] = {"/bin/bash", script, NULL};
    if (posix_spawn(&child, "/bin/bash", NULL, &attr, args, environ) != 0) return 1;
    signal(SIGTERM, forward); signal(SIGINT, forward); signal(SIGHUP, forward);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) { }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
