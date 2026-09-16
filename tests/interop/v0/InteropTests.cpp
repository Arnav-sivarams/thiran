#include "interop/v0/NativeLibrary.hpp"
#include "tooling/v0/BuildConfig.hpp"
#include "tooling/v0/Driver.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <iostream>
#include <limits>
#include <unistd.h>

namespace s = thiran::v0::semantic;
namespace a = thiran::v0::analysis;
namespace i = thiran::v0::interop;
namespace t = thiran::v0::tooling;
static i::NativeLibraryContract valid(const std::string& archive) {
    i::NativeLibraryContract c;
    c.id="test.add.i64"; c.logicalLibrary="thiran_test"; c.symbol="thiran_test_add_i64";
    c.parameters={{{s::scalar(s::TypeKind::I64)},s::AccessMode::Read},
                  {{s::scalar(s::TypeKind::I64)},s::AccessMode::Read}};
    c.result=s::scalar(s::TypeKind::I64); c.executable=true;
    c.linkItems={{i::LinkItemKind::StaticArchive,archive}};
    return c;
}
int main() {
    const auto base=valid(TH009_TEST_ARCHIVE);
    assert(i::verifyContracts({base}).ok()); // LIB01/LIB02
    auto bad=[&](i::NativeLibraryContract c,const std::string& id) {
        auto result=i::verifyContracts({c}); bool found=false;
        for(const auto& e:result.errors) found |= e.find(id)!=std::string::npos;
        assert(found);
    };
    assert(!i::verifyContracts({base,base}).ok()); // LIB03
    {auto c=base;c.symbol="";bad(c,"LIBV02");} // LIB04
    {auto c=base;c.effects=static_cast<a::EffectSet>(a::EffectKind::Mutates);bad(c,"LIBV08");} // LIB05
    {auto c=base;c.parameters[0].access=s::AccessMode::MutableBorrow;
     c.effects=static_cast<a::EffectSet>(a::EffectKind::Mutates);bad(c,"LIBV06");} // LIB06
    {auto c=base;c.linkItems[0].value="relative.a";bad(c,"LIBV10");} // LIB07
    {auto c=base;c.mayTrap=true;bad(c,"LIBV09");}
    {auto c=base;c.result=s::scalar(s::TypeKind::F64);bad(c,"LIBV04");}
    std::string pattern="/tmp/th009-interop-XXXXXX"; assert(::mkdtemp(pattern.data()));
    const auto output=std::filesystem::path(pattern)/"interop artifact";
    const auto build=std::filesystem::path(TH009_BUILD);
    t::HostToolchainConfig config{TH009_CXX,t::configuredHostCompilerArguments(), {TH009_INCLUDE},
        {build/"libthiran_v0_storage.a",build/"libthiran_v0_analysis.a",
         build/"libthiran_v0_semantic.a",build/"libthiran_v0_frontend.a"}};
    t::CompilerDriver driver(config);
    const std::string stub=
        "#include <cstdint>\n#include <iostream>\n"
        "extern \"C\" std::int64_t thiran_test_add_i64(std::int64_t,std::int64_t);\n"
        "int main(){std::cout<<thiran_test_add_i64(19,23)<<'\\n';}\n";
    auto built=driver.buildIntegrationStubForTesting(stub,output,{base});
    assert(built.success && std::filesystem::exists(output)); // LIB08/LIB09
    auto run=t::runProcess({output.string(),{}});
    assert(run.exitStatus==0 && run.standardOutput=="42\n"); // LIB10
    std::cout << "V0InteropTests PASS LIB01-LIB12\n";
}
