// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title EvidenceRegistry
 * @notice Tamper-evident, immutable digital evidence fingerprint registry for VeriFace Chain.
 * @dev Stores SHA-256 evidence hashes and registration timestamps.
 * 
 * CORE DESIGN PRINCIPLE:
 * The blockchain does NOT prove that a face belongs to someone.
 * The blockchain proves that a specific digital evidence fingerprint was registered
 * at a particular immutable point in time by an authorized uploader.
 */
contract EvidenceRegistry {

    /// @notice Data model representing an on-chain registered evidence record
    struct Record {
        bytes32 evidenceHash;
        uint256 timestamp;
        address uploader;
    }

    /// @notice Mapping from SHA-256 evidence hash (bytes32) to registered record
    mapping(bytes32 => Record) private _records;

    /// @notice Emitted when a new digital evidence fingerprint is notarized on chain
    event EvidenceRegistered(
        bytes32 indexed evidenceHash,
        uint256 timestamp,
        address indexed uploader
    );

    /// @notice Thrown when attempting to register a zero hash
    error InvalidZeroHash();

    /// @notice Thrown when attempting to register an evidence hash that already exists
    error EvidenceAlreadyRegistered(bytes32 evidenceHash, uint256 originalTimestamp);

    /// @notice Thrown when querying a non-existent evidence hash
    error EvidenceNotFound(bytes32 evidenceHash);

    /**
     * @notice Registers a new evidence fingerprint on the blockchain.
     * @param evidenceHash 32-byte SHA-256 digest of the canonicalized evidence record.
     */
    function registerEvidence(bytes32 evidenceHash) external {
        if (evidenceHash == bytes32(0)) {
            revert InvalidZeroHash();
        }

        Record memory existing = _records[evidenceHash];
        if (existing.timestamp != 0) {
            revert EvidenceAlreadyRegistered(evidenceHash, existing.timestamp);
        }

        _records[evidenceHash] = Record({
            evidenceHash: evidenceHash,
            timestamp: block.timestamp,
            uploader: msg.sender
        });

        emit EvidenceRegistered(evidenceHash, block.timestamp, msg.sender);
    }

    /**
     * @notice Verifies whether an evidence fingerprint exists on chain and returns its details.
     * @param evidenceHash 32-byte SHA-256 digest to verify.
     * @return exists True if the evidence is registered on chain.
     * @return timestamp Unix block timestamp when registered (0 if not found).
     * @return uploader Ethereum address of the entity that registered the evidence.
     */
    function verifyEvidence(bytes32 evidenceHash)
        external
        view
        returns (
            bool exists,
            uint256 timestamp,
            address uploader
        )
    {
        Record memory rec = _records[evidenceHash];
        if (rec.timestamp != 0) {
            return (true, rec.timestamp, rec.uploader);
        }
        return (false, 0, address(0));
    }

    /**
     * @notice Retrieves the full Record struct for a registered evidence hash.
     * @param evidenceHash 32-byte SHA-256 digest.
     * @return Complete Record struct.
     */
    function getEvidence(bytes32 evidenceHash) external view returns (Record memory) {
        Record memory rec = _records[evidenceHash];
        if (rec.timestamp == 0) {
            revert EvidenceNotFound(evidenceHash);
        }
        return rec;
    }
}
