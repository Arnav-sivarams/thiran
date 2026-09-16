#include "tooling/v0/Driver.hpp"
#include "tooling/v0/BuildConfig.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace fs=std::filesystem;
namespace t=thiran::v0::tooling;
static void write(const fs::path& path,const std::string& bytes) { std::ofstream f(path,std::ios::binary); f<<bytes; assert(f.good()); }
static t::HostToolchainConfig config() {
    fs::path build=TH009_BUILD;
    return {TH009_CXX,t::configuredHostCompilerArguments(), {TH009_INCLUDE},
            {build/"libthiran_v0_storage.a",build/"libthiran_v0_analysis.a",
             build/"libthiran_v0_semantic.a",build/"libthiran_v0_frontend.a"}};
}
static t::ProcessResult cli(std::vector<std::string> args) { return t::runProcess({TH009_CLI,std::move(args)}); }
int main() {
    std::string pattern="/tmp/th009 tooling XXXXXX"; assert(::mkdtemp(pattern.data()));
    fs::path dir=pattern;
    const std::string add="fn main() -> Tensor<i64,2> {\nlet A=[1,2;3,4]\nlet B=[5,6;7,8]\nreturn A + B\n}\n";
    const std::string matmul="fn main() -> Tensor<i64,2> {\nlet A=[1,2;3,4]\nreturn A * A\n}\n";
    const std::string f32="fn main(x: f32) -> f32 { return x*x }\n";
    fs::path supported=dir/"source with spaces.th", unsupported=dir/"matmul.th";
    fs::path meta=dir/"literal;dollar$(not-run).th";
    write(supported,add); write(unsupported,matmul); write(meta,add);
    t::CompilerDriver driver(config());
    auto checked=driver.checkSource({supported.string(),add}); assert(checked.success); // CLI02
    auto matChecked=driver.checkSource({unsupported.string(),matmul}); assert(matChecked.success); // CLI03
    auto f32Checked=driver.checkSource({"f32.th",f32}); assert(f32Checked.success);
    auto f32Build=driver.buildNative({"f32.th",f32},{"main",dir/"f32-bad",{}});
    assert(!f32Build.success && f32Build.stage==t::CompilerStage::Backend &&
           f32Build.coverage.find("fallback: NONE")!=std::string::npos);
    auto matBuild=driver.buildNative({unsupported.string(),matmul},{"main",dir/"bad",{}});
    assert(!matBuild.success && matBuild.stage==t::CompilerStage::Backend &&
           matBuild.coverage.find("MatMul unsupported-native")!=std::string::npos &&
           matBuild.coverage.find("fallback: NONE")!=std::string::npos); // CLI04/15
    auto region=driver.emitRegion({supported.string(),add},"main");
    assert(region.success && region.generatedSource==driver.emitRegion({supported.string(),add},"main").generatedSource);
    const std::string expectedRegion=
        "region f1 main\n"
        "  v1 Integer i64 {} =1\n  v2 Integer i64 {} =2\n"
        "  v3 Integer i64 {} =3\n  v4 Integer i64 {} =4\n"
        "  v5 TensorLiteral Tensor<i64,2> {2,2} v1 v2 v3 v4\n"
        "  v6 Integer i64 {} =5\n  v7 Integer i64 {} =6\n"
        "  v8 Integer i64 {} =7\n  v9 Integer i64 {} =8\n"
        "  v10 TensorLiteral Tensor<i64,2> {2,2} v6 v7 v8 v9\n"
        "  v11 Alias Tensor<i64,2> {2,2} v5\n"
        "  v12 Alias Tensor<i64,2> {2,2} v10\n"
        "  v13 Add Tensor<i64,2> {2,2} v11 v12\n  output v13\n";
    assert(*region.generatedSource==expectedRegion); // CLI05
    auto cpp=driver.emitNativeCpp({supported.string(),add},"main");
    assert(cpp.success && cpp.generatedSource==driver.emitNativeCpp({supported.string(),add},"main").generatedSource); // CLI06
    auto help=cli({"--help"}); assert(help.exitStatus==0 && help.standardOutput.find("experimental developer")!=std::string::npos); // CLI01
    auto check=cli({"check",supported.string()}); assert(check.exitStatus==0 && check.standardOutput=="CHECK PASS\n");
    auto checkUnsupported=cli({"check",unsupported.string()}); assert(checkUnsupported.exitStatus==0); // CLI03
    auto reject=cli({"build",unsupported.string(),"-o",(dir/"unsupported").string()});
    assert(reject.exitStatus!=0 && reject.standardError.find("BACKEND-UNSUPPORTED")!=std::string::npos &&
           reject.standardError.find("fallback: NONE")!=std::string::npos);
    auto emittedRegion=cli({"emit-region",supported.string()});
    assert(emittedRegion.exitStatus==0 && emittedRegion.standardOutput==expectedRegion);
    auto emittedCpp=cli({"emit-cpp",supported.string()});
    assert(emittedCpp.exitStatus==0 && emittedCpp.standardOutput==*cpp.generatedSource);
    fs::path output=dir/"output artifact with spaces";
    auto build=cli({"build",supported.string(),"-o",output.string()});
    assert(build.exitStatus==0 && fs::exists(output)); // CLI07/12/13
    auto artifact=t::runProcess({output.string(),{}});
    const std::string expected="{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[6,8,10,12]}\n";
    assert(artifact.exitStatus==0 && artifact.standardOutput==expected); // CLI08
    auto run=cli({"run",supported.string()}); assert(run.exitStatus==0 && run.standardOutput==expected); // CLI09
    fs::path syntax=dir/"syntax.th", semantic=dir/"semantic.th";
    write(syntax,"fn main( {\n"); write(semantic,"fn main() -> i64 { return missing }\n");
    auto syntaxRun=cli({"check",syntax.string()}); assert(syntaxRun.exitStatus!=0 && syntaxRun.standardError.find("syntax[")!=std::string::npos); // CLI10
    auto semanticRun=cli({"check",semantic.string()}); assert(semanticRun.exitStatus!=0 && semanticRun.standardError.find("semantic[")!=std::string::npos); // CLI11
    auto metaCheck=cli({"check",meta.string()}); assert(metaCheck.exitStatus==0); // CLI14 input literal
    fs::path metaOutput=dir/"artifact;dollar$(not-run)";
    auto metaBuild=cli({"build",meta.string(),"-o",metaOutput.string()});
    assert(metaBuild.exitStatus==0 && fs::exists(metaOutput)); // CLI14 output literal
    auto deps=t::runProcess({"ldd",{output.string()}});
    assert(deps.exitStatus==0 && deps.standardOutput.find("torch")==std::string::npos &&
           deps.standardOutput.find("python")==std::string::npos); // CLI16
    auto missing=config(); missing.compilerExecutable="/definitely/missing/th009-cxx";
    t::CompilerDriver badDriver(missing);
    auto failed=badDriver.buildNative({supported.string(),add},{"main",dir/"never",{}});
    assert(!failed.success && failed.stage==t::CompilerStage::HostCompiler);
    auto nonzero=t::runProcess({"/bin/false",{}}); assert(nonzero.launched && nonzero.exitStatus!=0);
    std::cout << "V0ToolingTests PASS CLI01-CLI16\n";
}
