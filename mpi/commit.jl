# The commit of a clone, through Julia's libgit2: the compute nodes may have no git command (JUPITER, once the jobs
# have purged the modules). Included by mpi/scaling.jl and by the precompile steps of the jobs.
using LibGit2

commit(dir) = try
    LibGit2.with(LibGit2.GitRepo(dir)) do repo
        c = LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))
        "$(string(LibGit2.GitHash(c))[1:7]) $(first(split(LibGit2.message(c), '\n')))"
    end
catch
    "unknown"
end
