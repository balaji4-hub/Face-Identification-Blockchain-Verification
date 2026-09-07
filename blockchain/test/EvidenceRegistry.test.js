const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("EvidenceRegistry Smart Contract", function () {
  let registry;
  let owner;
  let user1;
  let user2;

  // Test 32-byte hashes (SHA-256 equivalents)
  const sampleHash1 = ethers.keccak256(ethers.toUtf8Bytes("canonical_evidence_record_1"));
  const sampleHash2 = ethers.keccak256(ethers.toUtf8Bytes("canonical_evidence_record_2"));
  const zeroHash = ethers.ZeroHash;

  beforeEach(async function () {
    [owner, user1, user2] = await ethers.getSigners();
    const EvidenceRegistryFactory = await ethers.getContractFactory("EvidenceRegistry");
    registry = await EvidenceRegistryFactory.deploy();
    await registry.waitForDeployment();
  });

  it("Should register a valid evidence hash successfully and emit EvidenceRegistered event", async function () {
    const tx = await registry.connect(user1).registerEvidence(sampleHash1);
    await expect(tx)
      .to.emit(registry, "EvidenceRegistered")
      .withArgs(sampleHash1, (await ethers.provider.getBlock("latest")).timestamp, user1.address);
  });

  it("Should prevent duplicate registration of the same evidence hash", async function () {
    await registry.connect(user1).registerEvidence(sampleHash1);
    await expect(
      registry.connect(user2).registerEvidence(sampleHash1)
    ).to.be.revertedWithCustomError(registry, "EvidenceAlreadyRegistered");
  });

  it("Should reject invalid zero hash", async function () {
    await expect(
      registry.connect(user1).registerEvidence(zeroHash)
    ).to.be.revertedWithCustomError(registry, "InvalidZeroHash");
  });

  it("Should correctly verify an existing registered evidence record", async function () {
    await registry.connect(user1).registerEvidence(sampleHash1);
    const [exists, timestamp, uploader] = await registry.verifyEvidence(sampleHash1);

    expect(exists).to.be.true;
    expect(timestamp).to.be.gt(0);
    expect(uploader).to.equal(user1.address);
  });

  it("Should return exists = false for an unregistered evidence hash", async function () {
    const [exists, timestamp, uploader] = await registry.verifyEvidence(sampleHash2);
    expect(exists).to.be.false;
    expect(timestamp).to.equal(0);
    expect(uploader).to.equal(ethers.ZeroAddress);
  });

  it("Should retrieve full record via getEvidence", async function () {
    await registry.connect(user2).registerEvidence(sampleHash2);
    const record = await registry.getEvidence(sampleHash2);

    expect(record.evidenceHash).to.equal(sampleHash2);
    expect(record.uploader).to.equal(user2.address);
    expect(record.timestamp).to.be.gt(0);
  });

  it("Should revert getEvidence for unknown hash", async function () {
    await expect(
      registry.getEvidence(sampleHash1)
    ).to.be.revertedWithCustomError(registry, "EvidenceNotFound");
  });

  it("Should support multiple distinct unique evidence registrations", async function () {
    await registry.connect(user1).registerEvidence(sampleHash1);
    await registry.connect(user2).registerEvidence(sampleHash2);

    const [exists1] = await registry.verifyEvidence(sampleHash1);
    const [exists2] = await registry.verifyEvidence(sampleHash2);

    expect(exists1).to.be.true;
    expect(exists2).to.be.true;
  });
});
