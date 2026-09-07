const hre = require("hardhat");

async function main() {
  console.log("Deploying EvidenceRegistry to network:", hre.network.name);

  const EvidenceRegistry = await hre.ethers.getContractFactory("EvidenceRegistry");
  const registry = await EvidenceRegistry.deploy();

  await registry.waitForDeployment();
  const address = await registry.getAddress();

  console.log("EvidenceRegistry successfully deployed at address:", address);
  console.log("Update CONTRACT_ADDRESS in your .env with:", address);
}

main().catch((error) => {
  console.error("Deployment failed:", error);
  process.exitCode = 1;
});
