# Homebrew formula for llm-redact. Tap directly from this repo:
#   brew tap asanderson/llm-redact https://github.com/asanderson/llm-redact
#   brew install asanderson/llm-redact/llm-redact
#
# NOTE: url/sha256 below point at the llm-redact-proxy sdist on PyPI and are
# refreshed by scripts/update_formula.py as part of the release runbook
# (RELEASING.md). Until the first PyPI release lands they hold placeholders
# and the formula does not install.
class LlmRedact < Formula
  include Language::Python::Virtualenv

  desc "Privacy proxy that redacts secrets/PII from LLM API traffic and restores replies"
  homepage "https://github.com/asanderson/llm-redact"
  url "PLACEHOLDER_SDIST_URL"
  sha256 "PLACEHOLDER_SDIST_SHA256"
  license "AGPL-3.0-only"

  depends_on "python@3.13"

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/3b/72/5562aabb8dd7181e8e860622a38bea08d17842b99ecd4c91f84ac95251b0/anyio-4.14.1.tar.gz"
    sha256 "8d648a3544c1a700e3ff78615cd679e4c5c3f149904287e73687b2596963629e"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/c9/c7/424b75da314c1045981bd9777432fad05a9e0c69daa4ed7e308bbaffe405/certifi-2026.6.17.tar.gz"
    sha256 "024c88eeec92ca068db80f02b8b07c9cef7b9fe261d1d535abfd5abd6f6af432"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/76/d4/81420972a676e8ffea40450d8c8c92943e7218a78fe9b64359836cc9876b/click-8.4.2.tar.gz"
    sha256 "9a6cea6e60b17ebe0a44c5cc636d94f09bd66142c1cd7d8b4cd731c4917a15f6"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/01/ee/02a2c011bdab74c6fb3c75474d40b3052059d95df7e73351460c8588d963/h11-0.16.0.tar.gz"
    sha256 "4e35b956cf45792e4caa5885e69fba00bdbc6ffafbfa020300e549b208ee5ff1"
  end

  resource "httpcore" do
    url "https://files.pythonhosted.org/packages/06/94/82699a10bca87a5556c9c59b5963f2d039dbd239f25bc2a63907a05a14cb/httpcore-1.0.9.tar.gz"
    sha256 "6e34463af53fd2ab5d807f399a9b45ea31c3dfa2276f15a2c3f00afff6e176e8"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/b1/df/48c586a5fe32a0f01324ee087459e112ebb7224f646c0b5023f5e79e9956/httpx-0.28.1.tar.gz"
    sha256 "75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/cd/63/9496c57188a2ee585e0f1db071d75089a11e98aa86eb99d9d7618fc1edce/idna-3.18.tar.gz"
    sha256 "ffb385a7e039654cef1ab9ef32c6fafe283c0c0467bba1d9029738ce4a14a848"
  end

  resource "starlette" do
    url "https://files.pythonhosted.org/packages/eb/e3/7c1dc7381d9f8ab7d854328ebfa884e62cb3f3d8549ddfd37c7814f42afa/starlette-1.3.1.tar.gz"
    sha256 "05d0213193f2fbaae60e2ecb593b4add4262ad4e46536b54abe36f11a71724e0"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/9f/f6/cc9aadc0e481344a42095d222bfa764122fb8cfba708d1922917bd8bfb01/uvicorn-0.50.2.tar.gz"
    sha256 "b92bf03509b82bcb9d49e7335b4fd364518ad021c2dc18b4e6a2fec8c955a0bb"
  end

  def install
    virtualenv_install_with_resources
  end

  service do
    run [opt_bin/"llm-redact", "serve"]
    keep_alive true
    log_path var/"log/llm-redact.log"
    error_log_path var/"log/llm-redact.log"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/llm-redact --version")
  end
end
